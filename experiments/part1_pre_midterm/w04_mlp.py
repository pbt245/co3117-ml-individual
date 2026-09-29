"""W04 experiments: MLP trained with backpropagation on UCI HAR.

Run from the repo root:  .venv/bin/python -m experiments.part1_pre_midterm.w04_mlp
"""
from experiments.part1_pre_midterm.plotting import plot_confusion, plot_curves
from src.data import CLASS_NAMES, load_splits
from src.from_scratch.mlp import MLP
from src.metrics import clear_week, confusion_matrix, log_results, macro_f1, precision_recall_f1
from src.reference_adapters.sklearn_ref import sklearn_mlp

WEEK = "W04"
SEED = 0
EPOCHS = 40
HIDDEN = 128


def run(X_tr, y_tr, X_va, y_va, setting, **kwargs):
    net = MLP([X_tr.shape[1], HIDDEN, 6], seed=SEED, **kwargs)
    net.fit(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, batch_size=64)
    f1 = macro_f1(y_va, net.predict(X_va))
    log_results(WEEK, "mlp", setting, "val", y_va, net.predict(X_va))
    h = net.history
    best_epoch = min(range(len(h["val_loss"])), key=h["val_loss"].__getitem__) + 1
    print(f"  {setting:32s} val macroF1={f1:.4f}  train loss={h['train_loss'][-1]:.4f}"
          f"  val loss={h['val_loss'][-1]:.4f}  (min val loss {min(h['val_loss']):.4f} @ epoch {best_epoch})")
    return net


def main():
    clear_week(WEEK)
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(seed=SEED)

    print("[A] Controlled experiment: optimizer (ReLU, 1 hidden layer of 128)")
    settings = {
        "SGD lr=0.01": dict(optimizer="sgd", lr=0.01),
        "SGD+Momentum lr=0.01 mu=0.9": dict(optimizer="momentum", lr=0.01, momentum=0.9),
        "Adam lr=0.001": dict(optimizer="adam", lr=1e-3),
    }
    nets = {name: run(X_tr, y_tr, X_va, y_va, name, activation="relu", **kw)
            for name, kw in settings.items()}
    plot_curves({n: m.history["train_loss"] for n, m in nets.items()},
                "MLP training loss by optimizer", "cross-entropy", "w04_optimizer_train_loss.png", logy=True)
    plot_curves({n: m.history["val_loss"] for n, m in nets.items()},
                "MLP validation loss by optimizer", "cross-entropy", "w04_optimizer_val_loss.png")

    print("\n[B] Activation function (Adam lr=0.001)")
    for act in ["sigmoid", "tanh"]:
        run(X_tr, y_tr, X_va, y_va, f"Adam, {act}", activation=act, optimizer="adam", lr=1e-3)

    print("\n[C] Final model: ReLU + Adam + early stopping (patience=5) -> test set")
    final = MLP([X_tr.shape[1], HIDDEN, 6], activation="relu", optimizer="adam", lr=1e-3, seed=SEED)
    final.fit(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, batch_size=64, patience=5)
    y_hat = final.predict(X_te)
    acc, f1 = log_results(WEEK, "mlp", "relu+adam+early_stop", "test", y_te, y_hat)
    print(f"  stopped after {len(final.history['val_loss'])} epochs -> test acc={acc:.4f}  macroF1={f1:.4f}")
    for c, f in zip(CLASS_NAMES, precision_recall_f1(y_te, y_hat, 6)["f1"]):
        print(f"    F1[{c}] = {f:.4f}")
    plot_confusion(confusion_matrix(y_te, y_hat, 6), CLASS_NAMES,
                   "MLP (ReLU, Adam, early stop), test", "w04_mlp_confusion.png")

    ref = sklearn_mlp(X_tr, y_tr, hidden=(HIDDEN,), seed=SEED)
    acc, f1 = log_results(WEEK, "sklearn_mlp_ref", "relu+adam", "test", y_te, ref.predict(X_te))
    print(f"  sklearn MLPClassifier reference -> test acc={acc:.4f}  macroF1={f1:.4f}")


if __name__ == "__main__":
    main()
