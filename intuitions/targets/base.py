from abc import ABC, abstractmethod

import torch


class Target(ABC):
    """A fine-tuning target task: data splits, datasets, loss and metrics."""

    name: str
    label_names: tuple
    summary_metrics: tuple  # test metrics averaged over eval seeds; includes "macro_auc"
    eval_loss: torch.nn.Module

    @property
    def num_outputs(self):
        return len(self.label_names)

    def suggest_extra_hparams(self, trial):
        """Target-specific Optuna hyperparameters, suggested after the shared ones."""
        return {}

    @abstractmethod
    def split(self, seed):
        """Return (train_items, val_items, test_items) for a data seed."""

    @abstractmethod
    def make_dataset(self, items, hparams, train):
        """Dataset yielding (image in [0, 1], label) for the given items."""

    @abstractmethod
    def make_criterion(self, hparams):
        pass

    @abstractmethod
    def to_probs(self, logits):
        pass

    @abstractmethod
    def metrics(self, y_true, y_prob):
        """Metrics dict including "macro_auc"."""

    @abstractmethod
    def item_ids(self, items):
        """(ids, groups): a stable identifier per item and the unit (e.g. patient) to resample by."""

    @torch.no_grad()
    def predict(self, model, loader, device):
        """(y_true, y_prob, mean loss) over a loader, in loader order."""
        model.eval()
        labels, probs, total_loss = [], [], 0.0
        for images, y in loader:
            images, y = images.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                total_loss += self.eval_loss(logits, y).item()
            labels.append(y.cpu())
            probs.append(self.to_probs(logits.float()).cpu())
        return torch.cat(labels).numpy(), torch.cat(probs).numpy(), total_loss / max(len(loader), 1)

    def evaluate(self, model, loader, device):
        y_true, y_prob, loss = self.predict(model, loader, device)
        return {"loss": loss, **self.metrics(y_true, y_prob)}
