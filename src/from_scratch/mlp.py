"""W04: multilayer perceptron trained with backpropagation, written from scratch in NumPy.

Notation follows slides MLP.pdf / TrainingANN.pdf:
  h(0) = x,  z(l) = h(l-1) W(l)^T + b(l),  h(l) = phi(z(l)),  P = softmax(z(L+1)).
Weights have shape (out, in) like the slides' FC layer (Y = X W^T + 1 b^T).
"""
import numpy as np

from src.from_scratch.logistic import cross_entropy, softmax

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


ACTIVATIONS = {
    # name: (phi, phi')  -- derivative written in terms of z
    "relu": (lambda z: np.maximum(0.0, z), lambda z: (z > 0).astype(float)),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (_sigmoid, lambda z: _sigmoid(z) * (1.0 - _sigmoid(z))),
}


class MLP:
    def __init__(self, layer_sizes, activation="relu", optimizer="adam", lr=1e-3,
                 momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8, l2=0.0, seed=0):
        """layer_sizes = [n_in, hidden_1, ..., n_classes]."""
        self.rng = np.random.default_rng(seed)
        self.phi, self.dphi = ACTIVATIONS[activation]
        self.optimizer, self.lr, self.l2 = optimizer, lr, l2
        self.mu, self.beta1, self.beta2, self.eps = momentum, beta1, beta2, eps
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He init for ReLU, Xavier (Glorot) for sigmoid/tanh.
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(self.rng.normal(0.0, scale, size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]  # momentum / Adam 1st moment
        self.v = [np.zeros_like(p) for p in self.params]  # Adam 2nd moment
        self.t = 0

    # ---------- forward pass: compute and cache z(l), h(l) ----------
    def forward(self, X):
        self.cache_h, self.cache_z = [X], []
        h = X
        for l in range(len(self.W)):
            z = h @ self.W[l].T + self.b[l]
            self.cache_z.append(z)
            h = softmax(z) if l == len(self.W) - 1 else self.phi(z)
            self.cache_h.append(h)
        return h  # class probabilities P, shape (B, K)

    def loss(self, P, Y):
        reg = 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.W)
        return cross_entropy(Y, P) + reg

    # ---------- backward pass: chain rule from the output layer down ----------
    def backward(self, Y):
        B = Y.shape[0]
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        delta = (self.cache_h[-1] - Y) / B  # dL/dz(L+1) for softmax + cross-entropy
        for l in reversed(range(len(self.W))):
            grads_W[l] = delta.T @ self.cache_h[l] + self.l2 * self.W[l]
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                # delta(l) = (delta(l+1) W(l+1)) * phi'(z(l))
                delta = (delta @ self.W[l]) * self.dphi(self.cache_z[l - 1])
        return grads_W + grads_b

    # ---------- update step: SGD / SGD+Momentum / Adam ----------
    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.optimizer == "sgd":
                p -= self.lr * g
            elif self.optimizer == "momentum":
                self.m[i] = self.mu * self.m[i] + g
                p -= self.lr * self.m[i]
            elif self.optimizer == "adam":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                raise ValueError(self.optimizer)

    # ---------- SGD training loop (TrainingANN.pdf pseudocode) + early stopping ----------
    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=64, patience=None):
        K = self.W[-1].shape[0]
        Y = np.eye(K)[y]
        Y_val = np.eye(K)[y_val] if X_val is not None else None
        self.history = {"train_loss": [], "val_loss": []}
        best, best_params, wait = np.inf, None, 0
        for epoch in range(epochs):
            idx = self.rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                b = idx[start:start + batch_size]
                self.forward(X[b])
                self.step(self.backward(Y[b]))
            self.history["train_loss"].append(self.loss(self.forward(X), Y))
            if X_val is None:
                continue
            val_loss = self.loss(self.forward(X_val), Y_val)
            self.history["val_loss"].append(val_loss)
            if patience is not None:
                if val_loss < best:
                    best, wait = val_loss, 0
                    best_params = [p.copy() for p in self.params]
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_params is not None:  # restore the checkpoint with the lowest validation loss
            for p, bp in zip(self.params, best_params):
                p[...] = bp
        return self

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
