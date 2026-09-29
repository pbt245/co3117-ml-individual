"""W03 experiments: perceptron / delta rule / logistic regression / softmax regression on UCI HAR.

Run from the repo root:  .venv/bin/python -m experiments.part1_pre_midterm.w03_logreg
"""
import numpy as np

from experiments.part1_pre_midterm.plotting import plot_confusion, plot_curves
from src.data import CLASS_NAMES, load_splits
from src.from_scratch.logistic import DeltaRule, LogisticRegression, Perceptron, SoftmaxRegression
from src.metrics import clear_week, confusion_matrix, log_results, macro_f1, precision_recall_f1
from src.reference_adapters.sklearn_ref import sklearn_softmax_regression

WEEK = "W03"
SEED = 0


def binary_subset(X, y, a, b):
    mask = (y == a) | (y == b)
    return X[mask], (y[mask] == b).astype(int)


def part_a_binary(X_tr, y_tr, X_te, y_te):
    """Perceptron vs delta rule vs logistic regression on two binary pairs."""
    print("\n[A] Binary linear classifiers (test macro-F1)")
    pairs = {"WALKING-vs-LAYING": (0, 5), "SITTING-vs-STANDING": (3, 4)}
    mistakes = {}
    for name, (a, b) in pairs.items():
        Xa, ya = binary_subset(X_tr, y_tr, a, b)
        Xt, yt = binary_subset(X_te, y_te, a, b)
        models = {
            "perceptron": Perceptron(lr=0.1, epochs=20, seed=SEED),
            "delta_rule": DeltaRule(lr=0.001, epochs=20, seed=SEED),
            "logistic": LogisticRegression(lr=0.1, epochs=20, seed=SEED),
        }
        for mname, m in models.items():
            m.fit(Xa, ya)
            acc, f1 = log_results(WEEK, mname, name, "test", yt, m.predict(Xt))
            print(f"  {name:22s} {mname:11s} acc={acc:.4f}  macroF1={f1:.4f}")
        mistakes[name] = models["perceptron"].history
        print(f"  {name:22s} perceptron mistakes/epoch: {models['perceptron'].history}")
    plot_curves(mistakes, "Perceptron: training mistakes per epoch", "# mistakes",
                "w03_perceptron_mistakes.png")


def part_b_learning_rate(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Controlled experiment: effect of the learning rate on softmax regression."""
    print("\n[B] Softmax regression, learning-rate sweep (6 classes)")
    curves, results = {}, {}
    for lr in [0.001, 0.01, 0.1, 1.0]:
        m = SoftmaxRegression(n_classes=6, lr=lr, epochs=30, batch_size=64, seed=SEED)
        m.fit(X_tr, y_tr, X_va, y_va)
        va_f1 = macro_f1(y_va, m.predict(X_va))
        log_results(WEEK, "softmax_regression", f"lr={lr}", "val", y_va, m.predict(X_va))
        results[lr] = (va_f1, m)
        curves[f"lr={lr} train"] = m.history["train_loss"]
        print(f"  lr={lr:<6} val macroF1={va_f1:.4f}  final train loss={m.history['train_loss'][-1]:.4f}"
              f"  final val loss={m.history['val_loss'][-1]:.4f}")
    plot_curves(curves, "Softmax regression: training loss vs learning rate", "cross-entropy",
                "w03_softmax_lr_curves.png", logy=True)

    best_lr = max(results, key=lambda k: results[k][0])
    best = results[best_lr][1]
    y_hat = best.predict(X_te)
    acc, f1 = log_results(WEEK, "softmax_regression", f"best lr={best_lr}", "test", y_te, y_hat)
    print(f"  best lr={best_lr} -> test acc={acc:.4f}  macroF1={f1:.4f}")
    per_class = precision_recall_f1(y_te, y_hat, 6)["f1"]
    for c, f in zip(CLASS_NAMES, per_class):
        print(f"    F1[{c}] = {f:.4f}")
    plot_confusion(confusion_matrix(y_te, y_hat, 6), CLASS_NAMES,
                   f"Softmax regression (lr={best_lr}), test", "w03_softmax_confusion.png")

    ref = sklearn_softmax_regression(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    acc, f1 = log_results(WEEK, "sklearn_logreg_ref", "C=1.0", "test", y_te, ref.predict(X_te))
    print(f"  sklearn reference     -> test acc={acc:.4f}  macroF1={f1:.4f}")


def main():
    clear_week(WEEK)
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(seed=SEED)
    print(f"train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")
    part_a_binary(X_tr, y_tr, X_te, y_te)
    part_b_learning_rate(X_tr, y_tr, X_va, y_va, X_te, y_te)


if __name__ == "__main__":
    main()
