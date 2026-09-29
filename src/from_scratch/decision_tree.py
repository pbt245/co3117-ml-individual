"""W02: decision trees from scratch (slides: Decision_Tree.pdf).

- Impurity measures: entropy, Gini, information gain, gain ratio.
- ID3: multiway splits on categorical attributes by max information gain (Play Tennis sanity check).
- CART-style classifier: binary threshold splits on continuous features, Gini or entropy criterion,
  pre-pruning (max_depth, min_samples_split, min_impurity_decrease) and reduced-error post-pruning.
"""
from collections import Counter

import numpy as np


# ---------- impurity measures ----------
def entropy(counts):
    """H(S) = -sum_i p_i log2 p_i, computed from class counts."""
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def gini(counts):
    """Gini(S) = 1 - sum_i p_i^2."""
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def information_gain(parent_counts, children_counts, impurity=entropy):
    """IG(S, A) = H(S) - sum_v |S_v|/|S| H(S_v)  (with Gini this is the CART decrease Delta G)."""
    n = float(np.sum(parent_counts))
    weighted = sum(np.sum(c) / n * impurity(c) for c in children_counts)
    return impurity(parent_counts) - weighted


def split_info(children_counts):
    """SplitInfo(S, A) = -sum_v |S_v|/|S| log2(|S_v|/|S|)  (entropy of the partition sizes)."""
    return entropy([np.sum(c) for c in children_counts])


def gain_ratio(parent_counts, children_counts):
    """C4.5: GR(S, A) = IG(S, A) / SplitInfo(S, A)."""
    si = split_info(children_counts)
    return information_gain(parent_counts, children_counts) / si if si > 0 else 0.0


# ---------- ID3 on categorical data (used for the slide example) ----------
def _counts(labels, classes):
    c = Counter(labels)
    return [c[k] for k in classes]


def id3(rows, target, attributes, classes=None):
    """rows: list of dicts. Returns a nested dict {attr: {value: subtree}} or a class label (leaf)."""
    labels = [r[target] for r in rows]
    classes = classes or sorted(set(labels))
    if len(set(labels)) == 1:                      # pure node
        return labels[0]
    if not attributes:                             # no attribute left -> majority class
        return Counter(labels).most_common(1)[0][0]
    parent = _counts(labels, classes)
    gains = {}
    for a in attributes:
        values = sorted(set(r[a] for r in rows))
        children = [_counts([r[target] for r in rows if r[a] == v], classes) for v in values]
        gains[a] = information_gain(parent, children)
    best = max(attributes, key=gains.get)          # greedy: highest IG
    rest = [a for a in attributes if a != best]
    return {best: {v: id3([r for r in rows if r[best] == v], target, rest, classes)
                   for v in sorted(set(r[best] for r in rows))}}


# ---------- CART-style tree for continuous features ----------
class Node:
    __slots__ = ("feature", "threshold", "left", "right", "counts", "depth")

    def __init__(self, counts, depth):
        self.counts, self.depth = counts, depth
        self.feature = self.threshold = self.left = self.right = None

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def prediction(self):
        return int(np.argmax(self.counts))          # majority class at the node


class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def _impurity_vec(self, C):
        """Impurity of every row of a count matrix C (n_splits, K), vectorized."""
        n = C.sum(axis=1, keepdims=True)
        P = C / np.maximum(n, 1)
        if self.criterion == "gini":
            return 1.0 - np.sum(P ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.sum(np.where(P > 0, P * np.log2(P), 0.0), axis=1)

    def _best_split(self, X, Y1h):
        """Scan every feature and every midpoint threshold; return (gain, feature, threshold)."""
        n = len(X)
        parent_imp = self._impurity_vec(Y1h.sum(axis=0, keepdims=True))[0]
        best = (0.0, None, None)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            left = np.cumsum(Y1h[order], axis=0)[:-1]          # class counts of x <= xs[i]
            right = left[-1] + Y1h[order[-1]] - left            # remaining counts
            valid = xs[:-1] < xs[1:]                             # only cut between distinct values
            if not valid.any():
                continue
            n_left = np.arange(1, n)
            child = (n_left * self._impurity_vec(left) + (n - n_left) * self._impurity_vec(right)) / n
            gain = np.where(valid, parent_imp - child, -np.inf)  # Delta G(s, t) or IG
            i = int(np.argmax(gain))
            if gain[i] > best[0]:
                best = (float(gain[i]), j, (xs[i] + xs[i + 1]) / 2.0)
        return best

    def _grow(self, X, Y1h, depth):
        node = Node(Y1h.sum(axis=0), depth)
        pure = np.count_nonzero(node.counts) <= 1
        if pure or len(X) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return node                                          # pre-pruning stops
        gain, j, t = self._best_split(X, Y1h)
        if j is None or gain <= self.min_impurity_decrease:
            return node
        mask = X[:, j] <= t
        node.feature, node.threshold = j, t
        node.left = self._grow(X[mask], Y1h[mask], depth + 1)
        node.right = self._grow(X[~mask], Y1h[~mask], depth + 1)
        return node

    def fit(self, X, y, n_classes=None):
        self.K = n_classes or int(y.max()) + 1
        self.root = self._grow(X, np.eye(self.K)[y], depth=0)
        return self

    def _leaf(self, x, max_depth=None):
        node = self.root
        while not node.is_leaf and (max_depth is None or node.depth < max_depth):
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X, max_depth=None):
        """max_depth truncates the grown tree at prediction time. Because growth is greedy and
        top-down, this gives the same tree as training with that max_depth (pre-pruning)."""
        return np.array([self._leaf(x, max_depth).prediction for x in X])

    # ---------- reduced-error post-pruning (bottom-up, on a validation set) ----------
    def prune(self, X_val, y_val):
        """Replace a subtree by a majority leaf if that does not increase validation errors."""
        def walk(node, idx):
            if node.is_leaf:
                return int(np.sum(y_val[idx] != node.prediction))
            go_left = X_val[idx, node.feature] <= node.threshold
            err_subtree = walk(node.left, idx[go_left]) + walk(node.right, idx[~go_left])
            err_leaf = int(np.sum(y_val[idx] != node.prediction))
            if err_leaf <= err_subtree:                  # subtree not better -> collapse
                node.left = node.right = node.feature = node.threshold = None
                return err_leaf
            return err_subtree

        walk(self.root, np.arange(len(y_val)))
        return self

    # ---------- inspection ----------
    def n_leaves(self, node=None):
        node = node or self.root
        return 1 if node.is_leaf else self.n_leaves(node.left) + self.n_leaves(node.right)

    def depth(self, node=None):
        node = node or self.root
        return node.depth if node.is_leaf else max(self.depth(node.left), self.depth(node.right))

    def rules(self, feature_names=None, max_depth=2):
        """Human-readable top of the tree (interpretability)."""
        lines = []

        def walk(node, indent):
            if node.is_leaf or node.depth >= max_depth:
                lines.append(f"{indent}-> class {node.prediction}  (n={int(node.counts.sum())})")
                return
            name = feature_names[node.feature] if feature_names else f"x[{node.feature}]"
            lines.append(f"{indent}if {name} <= {node.threshold:.4f}:")
            walk(node.left, indent + "    ")
            lines.append(f"{indent}else:")
            walk(node.right, indent + "    ")

        walk(self.root, "")
        return "\n".join(lines)
