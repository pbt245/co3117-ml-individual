"""Decision tree checks: impurity numbers from the slides, ID3 root, and CART tree behaviour."""
import numpy as np

from src.from_scratch.decision_tree import (
    DecisionTreeClassifier, entropy, gain_ratio, gini, id3, information_gain,
)
from src.play_tennis import PLAY_TENNIS


def split_counts(attr):
    values = sorted(set(r[attr] for r in PLAY_TENNIS))
    return [[sum(1 for r in PLAY_TENNIS if r[attr] == v and r["Play"] == c) for c in ("No", "Yes")]
            for v in values]


def test_slide_numbers():
    parent = [5, 9]
    assert round(entropy(parent), 2) == 0.94
    assert round(gini(parent), 2) == 0.46
    assert entropy([0, 4]) == 0.0 and gini([2, 2]) == 0.5
    gains = {a: information_gain(parent, split_counts(a)) for a in ("Outlook", "Humidity", "Wind", "Temp")}
    assert [round(gains[a], 2) for a in ("Outlook", "Humidity", "Wind", "Temp")] == [0.25, 0.15, 0.05, 0.03]
    assert round(gain_ratio(parent, split_counts("Outlook")), 2) == 0.16


def test_id3_play_tennis_tree():
    tree = id3(PLAY_TENNIS, "Play", ["Outlook", "Temp", "Humidity", "Wind"])
    assert list(tree) == ["Outlook"]
    branches = tree["Outlook"]
    assert branches["Overcast"] == "Yes"
    assert list(branches["Sunny"]) == ["Humidity"]
    assert list(branches["Rain"]) == ["Wind"]


def test_cart_fits_training_data_and_respects_depth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0.5)).astype(int) + (X[:, 2] > 1).astype(int)
    full = DecisionTreeClassifier().fit(X, y)
    assert np.all(full.predict(X) == y)                  # unpruned tree memorises distinct points
    shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert shallow.depth() <= 2
    # truncating the full tree at prediction time == training with max_depth (greedy growth)
    assert np.all(full.predict(X, max_depth=2) == shallow.predict(X))


def test_pruning_never_increases_validation_error():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.8 * rng.normal(size=300) > 0).astype(int)   # noisy labels -> overfitting
    tree = DecisionTreeClassifier().fit(X[:200], y[:200])
    err_before = np.sum(tree.predict(X[200:]) != y[200:])
    leaves_before = tree.n_leaves()
    tree.prune(X[200:], y[200:])
    assert np.sum(tree.predict(X[200:]) != y[200:]) <= err_before
    assert tree.n_leaves() < leaves_before
