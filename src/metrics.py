"""Classification metrics implemented from scratch (W1: performance metrics & evaluation)."""
import csv
from datetime import date
from pathlib import Path

import numpy as np

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "metrics.csv"
CSV_FIELDS = ["date", "week", "model", "setting", "split", "accuracy", "macro_f1"]


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes=None):
    """C[i, j] = number of samples with true class i predicted as class j."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    C = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def precision_recall_f1(y_true, y_pred, n_classes=None):
    """Per-class precision, recall, F1 (one-vs-rest) and their macro averages."""
    C = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(C).astype(float)
    fp = C.sum(axis=0) - tp  # predicted as k but not k
    fn = C.sum(axis=1) - tp  # truly k but missed
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def macro_f1(y_true, y_pred, n_classes=None):
    return precision_recall_f1(y_true, y_pred, n_classes)["macro_f1"]


def clear_week(week, path=RESULTS_CSV):
    """Drop earlier rows of one week so re-running an experiment script does not duplicate them."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["week"] != week]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def log_results(week, model, setting, split, y_true, y_pred, path=RESULTS_CSV):
    """Append one row to results/metrics.csv and return (accuracy, macro_f1)."""
    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "date": date.today().isoformat(), "week": week, "model": model, "setting": setting,
            "split": split, "accuracy": f"{acc:.4f}", "macro_f1": f"{f1:.4f}",
        })
    return acc, f1
