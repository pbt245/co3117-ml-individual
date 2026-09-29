"""Finite-difference gradient checks: analytic backprop must match numerical derivatives."""
import numpy as np

from src.from_scratch.logistic import SoftmaxRegression, add_bias
from src.from_scratch.mlp import MLP


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_softmax_regression_gradient():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 5)), rng.integers(0, 3, size=20)
    model = SoftmaxRegression(n_classes=3, l2=0.1)
    model.W = rng.normal(size=(6, 3))
    analytic = model.gradient(add_bias(X), np.eye(3)[y])
    numeric = np.zeros_like(model.W)
    h = 1e-6
    for idx in np.ndindex(model.W.shape):
        old = model.W[idx]
        model.W[idx] = old + h
        lp = model.loss(X, y)
        model.W[idx] = old - h
        lm = model.loss(X, y)
        model.W[idx] = old
        numeric[idx] = (lp - lm) / (2 * h)
    assert rel_error(analytic, numeric) < 1e-6


def check_mlp(activation):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(8, 4)), rng.integers(0, 3, size=8)
    Y = np.eye(3)[y]
    net = MLP([4, 5, 3], activation=activation, l2=0.01, seed=2)
    net.forward(X)
    analytic = net.backward(Y)
    h = 1e-6
    for p, g in zip(net.params, analytic):
        numeric = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + h
            lp = net.loss(net.forward(X), Y)
            p[idx] = old - h
            lm = net.loss(net.forward(X), Y)
            p[idx] = old
            numeric[idx] = (lp - lm) / (2 * h)
        assert rel_error(g, numeric) < 1e-5


def test_mlp_gradient_tanh():
    check_mlp("tanh")


def test_mlp_gradient_sigmoid():
    check_mlp("sigmoid")


def test_mlp_gradient_relu():
    check_mlp("relu")
