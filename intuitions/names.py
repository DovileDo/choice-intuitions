"""Display names of sources, targets and metrics in figures and tables."""

# The survey's source order (ImageNet-1K, RadImageNet, Ecoset), then the benchmark's additions.
SOURCE_NAMES = {"imagenet": "ImageNet-1K", "radimagenet": "RadImageNet", "ecoset_baseline": "Ecoset",
                "ecoset_dvd_s": "Ecoset DVD-S", "scratch": "Random init."}
TARGET_NAMES = {"crc": "CS-tissue", "starc9": "CS-tissue (STARC-9)", "chexpert": "CS-xray"}
METRIC_NAMES = {"macro_auc": "Test macro-AUC", "balanced_accuracy": "Test balanced accuracy",
                "macro_f1": "Test macro-F1"}
