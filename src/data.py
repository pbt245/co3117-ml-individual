"""Data loading and preprocessing for the UCI Human Activity Recognition (HAR) dataset.

Use case: classify 6 daily activities from 561 hand-crafted smartphone sensor features.
Raw files are expected in data/raw/UCI HAR Dataset/ (see data/README.md).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAR_DIR = ROOT / "data" / "raw" / "UCI HAR Dataset"

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_har(split="train", data_dir=HAR_DIR):
    """Return (X, y, subject) for split in {"train", "test"}.

    X: (N, 561) float features, y: (N,) int labels in 0..5, subject: (N,) subject ids.
    """
    d = Path(data_dir) / split
    if not d.exists():
        raise FileNotFoundError(f"{d} not found. Download the dataset first (see data/README.md).")
    X = np.loadtxt(d / f"X_{split}.txt")
    y = np.loadtxt(d / f"y_{split}.txt", dtype=int) - 1  # labels 1..6 -> 0..5
    subject = np.loadtxt(d / f"subject_{split}.txt", dtype=int)
    return X, y, subject


def load_feature_names(data_dir=HAR_DIR):
    """The 561 feature names from features.txt (e.g. 'tBodyAcc-mean()-X')."""
    with open(Path(data_dir) / "features.txt") as f:
        return [line.split(maxsplit=1)[1].strip() for line in f]


def train_val_split(X, y, subject, val_frac=0.2, seed=0):
    """Split by subject (not by row) so the same person never appears in both train and val.

    Rows from one subject are strongly correlated; a random row split would leak information
    and give an optimistic validation score.
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject)
    n_val = max(1, int(round(val_frac * len(subjects))))
    val_subjects = rng.choice(subjects, size=n_val, replace=False)
    val_mask = np.isin(subject, val_subjects)
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]


def standardize(X_train, *others):
    """Z-score using mean/std of the training set only (no test statistics leak into training)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    out = [(X_train - mu) / sd] + [(X - mu) / sd for X in others]
    return out if others else out[0]


def one_hot(y, n_classes=len(CLASS_NAMES)):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def load_splits(seed=0):
    """Convenience: standardized train / val / test arrays used by all experiments."""
    X_tr_full, y_tr_full, s_tr = load_har("train")
    X_te, y_te, _ = load_har("test")
    X_tr, y_tr, X_va, y_va = train_val_split(X_tr_full, y_tr_full, s_tr, seed=seed)
    X_tr, X_va, X_te = standardize(X_tr, X_va, X_te)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
