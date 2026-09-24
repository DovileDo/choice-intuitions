"""
Shared fine-tuning protocol.

Phase 1: Optuna TPE search (60 trials); each trial trains on every search seed
(0-4) and is scored by the mean best validation macro AUC. The randomly
initialised baseline gets its own search space and schedule (see REGIMES).
Phase 2: retrain with the best hyperparameters on held-out eval seeds (5-9)
and report test metrics as mean +/- std. Each eval seed is saved on its own
(metrics and per-image test predictions), so more seeds can be added later and
sources can be compared statistically afterwards.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from intuitions.source_models import is_pretrained, load_source_model
from intuitions.utils import set_seed

log = logging.getLogger(__name__)

SEARCH_SEEDS = (0, 1, 2, 3, 4)
EVAL_SEEDS = (5, 6, 7, 8, 9)
N_TRIALS = 60
NUM_WORKERS = 4

# Training from random initialisation needs higher learning rates and longer
# schedules than fine-tuning (He et al. 2019, "Rethinking ImageNet Pre-training").
REGIMES = {
    "pretrained": {"backbone_lr": (1e-5, 1e-3), "head_lr": (1e-4, 1e-2), "weight_decay": (1e-5, 1e-2),
                   "max_epochs": [20, 30, 50], "early_stop_patience": 7, "warmup_epochs": 2},
    "scratch": {"backbone_lr": (1e-4, 1e-2), "head_lr": (1e-4, 1e-2), "weight_decay": (1e-4, 1e-1),
                "max_epochs": [50, 100, 150], "early_stop_patience": 15, "warmup_epochs": 5},
}


def regime(source):
    return REGIMES["pretrained" if is_pretrained(source) else "scratch"]


@dataclass
class RunResult:
    best_val_auc: float
    history: list
    test_metrics: Optional[dict] = None
    test_labels: Optional[np.ndarray] = None  # in test_items order
    test_probs: Optional[np.ndarray] = None


def suggest_hparams(trial, target, source):
    space = regime(source)
    hparams = {
        "backbone_lr": trial.suggest_float("backbone_lr", *space["backbone_lr"], log=True),
        "head_lr": trial.suggest_float("head_lr", *space["head_lr"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.2, 0.5]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs": trial.suggest_categorical("max_epochs", space["max_epochs"]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }
    hparams.update(target.suggest_extra_hparams(trial))
    return hparams


def get_optimizer(model, backbone_lr, head_lr, weight_decay):
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)


def get_scheduler(optimizer, max_epochs, warmup_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(max_epochs - warmup_epochs, 1))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)


def train_model(target, source, train_items, val_items, hparams, device, test_items=None):
    """
    Fine-tune `source` on `target`, keeping the epoch with the best validation
    macro AUC (early stopping on it). If test_items is given, that epoch's model
    is evaluated on them.
    """
    set_seed(hparams["training_seed"])
    bs = hparams["batch_size"]
    train_loader = _loader(target.make_dataset(train_items, hparams, train=True), bs, shuffle=True)
    val_loader = _loader(target.make_dataset(val_items, hparams, train=False), bs, shuffle=False)

    fixed = regime(source)
    model = load_source_model(source, target.num_outputs, dropout=hparams["dropout"]).to(device)
    optimizer = get_optimizer(model, hparams["backbone_lr"], hparams["head_lr"], hparams["weight_decay"])
    scheduler = get_scheduler(optimizer, hparams["max_epochs"], fixed["warmup_epochs"])
    criterion = target.make_criterion(hparams)
    scaler = torch.amp.GradScaler()

    best_val_auc, best_state, patience, history = 0.0, None, 0, []
    for epoch in range(1, hparams["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        val = target.evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_macro_auc": val["macro_auc"]})
        log.debug("seed %d epoch %d/%d: train_loss %.4f val_loss %.4f val_macro_auc %.4f",
                  hparams["training_seed"], epoch, hparams["max_epochs"], train_loss, val["loss"], val["macro_auc"])

        if val["macro_auc"] > best_val_auc:
            best_val_auc = val["macro_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= fixed["early_stop_patience"]:
                log.debug("early stop at epoch %d", epoch)
                break

    result = RunResult(best_val_auc, history)
    if test_items is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_loader = _loader(target.make_dataset(test_items, hparams, train=False), bs, shuffle=False)
        result.test_labels, result.test_probs, test_loss = target.predict(model, test_loader, device)
        result.test_metrics = {"loss": test_loss, **target.metrics(result.test_labels, result.test_probs)}
    return result


def _objective(trial, target, source, device, seeds):
    hparams = suggest_hparams(trial, target, source)
    log.info("trial %d: %s", trial.number, hparams)
    val_aucs = []
    for step, seed in enumerate(seeds):
        train_items, val_items, _ = target.split(seed)
        val_auc = train_model(target, source, train_items, val_items,
                              {**hparams, "training_seed": seed}, device).best_val_auc
        val_aucs.append(val_auc)
        log.info("trial %d seed %d: best val macro AUC %.4f", trial.number, seed, val_auc)
        trial.report(float(np.mean(val_aucs)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(val_aucs))


def run_search(target, source, out_dir, device, n_trials=N_TRIALS, seeds=SEARCH_SEEDS):
    """Run (or resume, topping up to n_trials) the Optuna search; returns the best hyperparameters."""
    study = optuna.create_study(
        study_name=f"resnet50_{target.name}_{source}",
        storage=f"sqlite:///{out_dir / 'optuna_study.db'}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    finished = sum(t.state in finished_states for t in study.trials)
    if finished:
        log.info("Resuming study with %d/%d finished trials", finished, n_trials)
    study.optimize(lambda trial: _objective(trial, target, source, device, seeds),
                   n_trials=max(n_trials - finished, 0))

    log.info("Best trial %d: mean val macro AUC %.4f, params %s",
             study.best_trial.number, study.best_value, study.best_params)
    (out_dir / "best_hparams.json").write_text(json.dumps(study.best_params, indent=2))
    return study.best_params


def save_predictions(path, target, test_items, run):
    ids, groups = target.item_ids(test_items)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=run.test_labels, y_prob=run.test_probs, ids=np.array(ids),
                        groups=np.array(groups), label_names=np.array(target.label_names))


def _write_json(path, obj):
    # Write, then rename: an interrupted run never leaves a half-written file that looks finished.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _seed_record_path(out_dir, seed):
    return out_dir / "seeds" / f"seed_{seed}.json"


def _import_single_file_results(out_dir):
    """Split a results.json written before per-seed files existed into seeds/seed_<N>.json."""
    results_path = out_dir / "results.json"
    if (out_dir / "seeds").exists() or not results_path.exists():
        return
    results = json.loads(results_path.read_text())
    for record in results["per_seed_results"]:
        _write_json(_seed_record_path(out_dir, record["seed"]), {**record, "hparams": results["best_hparams"]})


def _load_seed_records(out_dir, best_hparams):
    """Finished eval seeds by seed number; all must have been trained with best_hparams."""
    _import_single_file_results(out_dir)
    records = {}
    for path in sorted((out_dir / "seeds").glob("seed_*.json")):
        record = json.loads(path.read_text())
        if record["hparams"] != best_hparams:
            raise RuntimeError(f"{path} was trained with {record['hparams']}, not the current best hyperparameters "
                               f"{best_hparams}; move the old seeds/ and predictions/ aside to start the eval over")
        records[record["seed"]] = record
    return records


def run_final_eval(target, source, out_dir, best_hparams, device, seeds=EVAL_SEEDS):
    """
    Train and test with best_hparams on every eval seed not finished yet, then
    summarise all finished seeds (including those from earlier invocations) in
    results.json. Extending a finished eval with more seeds only runs the new ones.
    """
    records = _load_seed_records(out_dir, best_hparams)
    for seed in seeds:
        if seed in records:
            log.info("Eval seed %d already finished, skipping", seed)
            continue
        train_items, val_items, test_items = target.split(seed)
        log.info("Eval seed %d: train %d, val %d, test %d", seed, len(train_items), len(val_items), len(test_items))
        run = train_model(target, source, train_items, val_items, {**best_hparams, "training_seed": seed}, device,
                          test_items=test_items)
        pred_path = out_dir / "predictions" / f"seed_{seed}.npz"
        save_predictions(pred_path, target, test_items, run)
        log.info("Eval seed %d: val macro AUC %.4f | test %s", seed, run.best_val_auc,
                 ", ".join(f"{m} {run.test_metrics[m]:.4f}" for m in target.summary_metrics))
        records[seed] = {"seed": seed, "hparams": best_hparams, "n_train": len(train_items),
                         "n_val": len(val_items), "n_test": len(test_items), "val_auc": run.best_val_auc,
                         "test_metrics": run.test_metrics, "predictions": str(pred_path.relative_to(out_dir)),
                         "history": run.history}
        _write_json(_seed_record_path(out_dir, seed), records[seed])

    per_seed = [records[s] for s in sorted(records)]
    summary = {}
    for m in target.summary_metrics:
        values = [r["test_metrics"][m] for r in per_seed]
        summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        log.info("FINAL %s over %d seeds: %.4f +/- %.4f", m, len(values), summary[m]["mean"], summary[m]["std"])

    results = {"target": target.name, "source": source, "best_hparams": best_hparams,
               "eval_seeds": sorted(records), "per_seed_results": per_seed, "summary": summary}
    _write_json(out_dir / "results.json", results)
    return results
