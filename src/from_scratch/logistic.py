"""W03: linear classifiers from scratch (perceptron, delta rule, logistic & softmax regression).

All models absorb the bias by feature augmentation x~ = [1, x] (slides: LogisticRegression.pdf, p.3).
"""
import numpy as np


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    # Numerically stable: never evaluates exp of a large positive number.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # shift-invariance trick: avoids overflow
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def bce(y, p, eps=1e-12):
    """Binary cross-entropy = mean negative log-likelihood of a Bernoulli model."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cross_entropy(Y, P, eps=1e-12):
    """Categorical cross-entropy for one-hot Y and probabilities P."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, eps, 1.0)), axis=1)))


def _minibatches(n, batch_size, rng):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


class Perceptron:
    """Rosenblatt perceptron, labels y in {0,1}. Updates only on mistakes:
    w <- w + eta * (y - y_hat) * x, with y_hat = step(w^T x).
    """

    def __init__(self, lr=0.1, epochs=20, seed=0):
        self.lr, self.epochs, self.seed = lr, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xb = add_bias(X)
        self.w = np.zeros(Xb.shape[1])
        self.history = []
        for _ in range(self.epochs):
            mistakes = 0
            for i in rng.permutation(len(y)):
                y_hat = 1 if Xb[i] @ self.w >= 0 else 0
                if y_hat != y[i]:
                    self.w += self.lr * (y[i] - y_hat) * Xb[i]
                    mistakes += 1
            self.history.append(mistakes)
        return self

    def predict(self, X):
        return (add_bias(X) @ self.w >= 0).astype(int)


class DeltaRule:
    """ADALINE / Widrow-Hoff delta rule: linear output, squared error, gradient descent.
    Batch update: w <- w - eta * X^T (Xw - y) / B  (same gradient as linear regression, W03).
    Targets are mapped to {-1, +1}; prediction thresholds the linear output at 0.
    """

    def __init__(self, lr=0.01, epochs=50, batch_size=32, seed=0):
        self.lr, self.epochs, self.batch_size, self.seed = lr, epochs, batch_size, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xb, t = add_bias(X), 2.0 * y - 1.0
        self.w = np.zeros(Xb.shape[1])
        self.history = []
        for _ in range(self.epochs):
            for b in _minibatches(len(t), self.batch_size, rng):
                err = Xb[b] @ self.w - t[b]
                self.w -= self.lr * Xb[b].T @ err / len(b)
            self.history.append(float(np.mean((Xb @ self.w - t) ** 2)))
        return self

    def predict(self, X):
        return (add_bias(X) @ self.w >= 0).astype(int)


class LogisticRegression:
    """Binary logistic regression: y_hat = sigmoid(w^T x), loss = BCE.
    Gradient (slides eq. 3.9): dL/dw = X^T (y_hat - y) / B.
    """

    def __init__(self, lr=0.1, epochs=50, batch_size=32, l2=0.0, seed=0):
        self.lr, self.epochs, self.batch_size, self.l2, self.seed = lr, epochs, batch_size, l2, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xb = add_bias(X)
        self.w = np.zeros(Xb.shape[1])
        self.history = []
        for _ in range(self.epochs):
            for b in _minibatches(len(y), self.batch_size, rng):
                p = sigmoid(Xb[b] @ self.w)
                grad = Xb[b].T @ (p - y[b]) / len(b) + self.l2 * self.w
                self.w -= self.lr * grad
            self.history.append(bce(y, sigmoid(Xb @ self.w)))
        return self

    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    """Multiclass logistic regression: P = softmax(X W), loss = categorical cross-entropy.
    Gradient: dL/dW = X^T (P - Y) / B  (+ l2 * W, bias row not regularized).
    """

    def __init__(self, n_classes, lr=0.1, epochs=50, batch_size=64, l2=0.0, seed=0):
        self.K, self.lr, self.epochs = n_classes, lr, epochs
        self.batch_size, self.l2, self.seed = batch_size, l2, seed

    def loss(self, X, y):
        Xb = add_bias(X)
        Y = np.eye(self.K)[y]
        reg = 0.5 * self.l2 * np.sum(self.W[1:] ** 2)
        return cross_entropy(Y, softmax(Xb @ self.W)) + reg

    def gradient(self, Xb, Y):
        P = softmax(Xb @ self.W)
        grad = Xb.T @ (P - Y) / len(Y)
        grad[1:] += self.l2 * self.W[1:]
        return grad

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        Xb, Y = add_bias(X), np.eye(self.K)[y]
        self.W = np.zeros((Xb.shape[1], self.K))
        self.history = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            for b in _minibatches(len(y), self.batch_size, rng):
                self.W -= self.lr * self.gradient(Xb[b], Y[b])
            self.history["train_loss"].append(self.loss(X, y))
            if X_val is not None:
                self.history["val_loss"].append(self.loss(X_val, y_val))
        return self

    def predict_proba(self, X):
        return softmax(add_bias(X) @ self.W)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
