"""W02 experiments: decision trees (ID3 sanity check + CART-style tree on UCI HAR).

Run from the repo root:  .venv/bin/python -m experiments.part1_pre_midterm.w02_decision_tree
"""
import copy
import time

from experiments.part1_pre_midterm.plotting import plot_confusion, plot_curves
from src.data import CLASS_NAMES, load_feature_names, load_splits
from src.from_scratch.decision_tree import (
    DecisionTreeClassifier, entropy, gain_ratio, gini, id3, information_gain,
)
from src.metrics import clear_week, confusion_matrix, log_results, macro_f1, precision_recall_f1
from src.play_tennis import PLAY_TENNIS
from src.reference_adapters.sklearn_ref import sklearn_decision_tree

WEEK = "W02"
SEED = 0
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]


def part_a_play_tennis():
    """Reproduce the slide numbers (entropy, Gini, IG, gain ratio) and the ID3 tree."""
    print("[A] Play Tennis (slides)")
    parent = [5, 9]
    print(f"  H(S) = {entropy(parent):.3f}   Gini(S) = {gini(parent):.3f}")
    for attr in ["Outlook", "Humidity", "Wind", "Temp"]:
        values = sorted(set(r[attr] for r in PLAY_TENNIS))
        children = [[sum(1 for r in PLAY_TENNIS if r[attr] == v and r["Play"] == c) for c in ("No", "Yes")]
                    for v in values]
        print(f"  {attr:9s} IG = {information_gain(parent, children):.3f}"
              f"   GR = {gain_ratio(parent, children):.3f}"
              f"   Gini decrease = {information_gain(parent, children, impurity=gini):.3f}")
    print(f"  ID3 tree: {id3(PLAY_TENNIS, 'Play', ['Outlook', 'Temp', 'Humidity', 'Wind'])}")


def main():
    clear_week(WEEK)
    part_a_play_tennis()
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_splits(seed=SEED)
    names = load_feature_names()

    print("\n[B] Controlled experiment: max_depth (pre-pruning), criterion = Gini vs entropy")
    trees, curves = {}, {}
    for crit in ["gini", "entropy"]:
        t0 = time.time()
        tree = DecisionTreeClassifier(criterion=crit).fit(X_tr, y_tr, n_classes=6)
        trees[crit] = tree
        print(f"  full {crit} tree: depth={tree.depth()}, leaves={tree.n_leaves()}, "
              f"fit {time.time() - t0:.1f}s")
        tr_f1, va_f1 = [], []
        for d in DEPTHS + [None]:
            tr_f1.append(macro_f1(y_tr, tree.predict(X_tr, max_depth=d)))
            va_f1.append(macro_f1(y_va, tree.predict(X_va, max_depth=d)))
            log_results(WEEK, f"tree_{crit}", f"max_depth={d}", "val", y_va, tree.predict(X_va, max_depth=d))
            print(f"    max_depth={str(d):4s} train macroF1={tr_f1[-1]:.4f}  val macroF1={va_f1[-1]:.4f}")
        curves[f"{crit} train"] = tr_f1
        curves[f"{crit} val"] = va_f1
    plot_curves(curves, "Decision tree: macro-F1 vs max_depth (last point = unlimited)", "macro-F1",
                "w02_tree_depth_curve.png")

    # model selection on validation only
    best_crit, best_d, best_f1 = None, None, -1.0
    for crit, tree in trees.items():
        for d in DEPTHS + [None]:
            f1 = macro_f1(y_va, tree.predict(X_va, max_depth=d))
            if f1 > best_f1:
                best_crit, best_d, best_f1 = crit, d, f1
    print(f"  selected on val: criterion={best_crit}, max_depth={best_d} (val macroF1={best_f1:.4f})")

    print("\n[C] Reduced-error post-pruning of the full Gini tree (on the validation set)")
    pruned = copy.deepcopy(trees["gini"])
    leaves_before = pruned.n_leaves()
    pruned.prune(X_va, y_va)
    print(f"  leaves {leaves_before} -> {pruned.n_leaves()}, depth {trees['gini'].depth()} -> {pruned.depth()}")
    print("  (validation was used for pruning, so only the test score is a fair estimate)")

    print("\n[D] Test set")
    y_hat = trees[best_crit].predict(X_te, max_depth=best_d)
    acc, f1 = log_results(WEEK, f"tree_{best_crit}", f"max_depth={best_d} (selected)", "test", y_te, y_hat)
    print(f"  pre-pruned tree ({best_crit}, depth {best_d}): test acc={acc:.4f}  macroF1={f1:.4f}")
    for c, f in zip(CLASS_NAMES, precision_recall_f1(y_te, y_hat, 6)["f1"]):
        print(f"    F1[{c}] = {f:.4f}")
    plot_confusion(confusion_matrix(y_te, y_hat, 6), CLASS_NAMES,
                   f"Decision tree ({best_crit}, depth {best_d}), test", "w02_tree_confusion.png")

    acc, f1 = log_results(WEEK, "tree_gini", "full, unpruned", "test", y_te, trees["gini"].predict(X_te))
    print(f"  full unpruned Gini tree:        test acc={acc:.4f}  macroF1={f1:.4f}")
    acc, f1 = log_results(WEEK, "tree_gini", "reduced-error pruned", "test", y_te, pruned.predict(X_te))
    print(f"  post-pruned Gini tree:          test acc={acc:.4f}  macroF1={f1:.4f}")

    ref = sklearn_decision_tree(X_tr, y_tr, criterion=best_crit, max_depth=best_d, seed=SEED)
    acc, f1 = log_results(WEEK, "sklearn_tree_ref", f"{best_crit}, max_depth={best_d}", "test", y_te,
                          ref.predict(X_te))
    print(f"  sklearn DecisionTreeClassifier: test acc={acc:.4f}  macroF1={f1:.4f}")

    print("\n[E] Top of the selected tree (interpretability); classes:",
          ", ".join(f"{i}={c}" for i, c in enumerate(CLASS_NAMES)))
    print(trees[best_crit].rules(names, max_depth=2))


if __name__ == "__main__":
    main()
