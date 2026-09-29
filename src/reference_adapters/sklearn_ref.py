"""Reference implementations from scikit-learn.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def sklearn_softmax_regression(X, y, C=1.0, seed=0):
    return LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X, y)


def sklearn_mlp(X, y, hidden=(128,), activation="relu", solver="adam", lr=1e-3, seed=0):
    return MLPClassifier(hidden_layer_sizes=hidden, activation=activation, solver=solver,
                         learning_rate_init=lr, max_iter=200, early_stopping=True,
                         random_state=seed).fit(X, y)


def sklearn_decision_tree(X, y, criterion="gini", max_depth=None, seed=0):
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=seed).fit(X, y)
