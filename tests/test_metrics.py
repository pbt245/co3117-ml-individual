"""From-scratch metrics must agree with scikit-learn."""
import numpy as np
from sklearn import metrics as skm

from src.metrics import accuracy, confusion_matrix, precision_recall_f1


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 6, 300), rng.integers(0, 6, 300)
    assert np.isclose(accuracy(y_true, y_pred), skm.accuracy_score(y_true, y_pred))
    assert (confusion_matrix(y_true, y_pred, 6) == skm.confusion_matrix(y_true, y_pred)).all()
    r = precision_recall_f1(y_true, y_pred, 6)
    assert np.isclose(r["macro_f1"], skm.f1_score(y_true, y_pred, average="macro"))
    assert np.allclose(r["precision"], skm.precision_score(y_true, y_pred, average=None))
    assert np.allclose(r["recall"], skm.recall_score(y_true, y_pred, average=None))
