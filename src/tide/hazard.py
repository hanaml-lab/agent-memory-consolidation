"""Discrete-time hazard model of memory access (Eqs. 3-4)."""
from __future__ import annotations
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class HazardModel:
    """h(k | x) = sigmoid(MLP(x) . e_k). A one-hidden-layer MLP trained with Adam in numpy."""

    def __init__(self, d_in: int, max_k: int = 64, hidden: int = 64, d_emb: int = 16, lr: float = 3e-3, seed: int = 0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(d_in), (d_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, d_emb)); self.b2 = np.zeros(d_emb)
        self.E = r.normal(0, 0.1, (max_k + 1, d_emb)); self.E[:, 0] = -2.0  # bias channel: low prior hazard
        self.max_k, self.lr, self.t = max_k, lr, 0
        self._m = {k: 0 for k in self._params()}; self._v = {k: 0 for k in self._params()}

    def _params(self):
        return ["W1", "b1", "W2", "b2", "E"]

    def _embed(self, X):
        H = np.maximum(X @ self.W1 + self.b1, 0)
        return H, H @ self.W2 + self.b2

    def hazard(self, X: np.ndarray, ks: np.ndarray | None = None) -> np.ndarray:
        """Return h of shape (n, K) for k = 1..max_k (or the given ks)."""
        _, Z = self._embed(np.atleast_2d(X))
        ks = np.arange(1, self.max_k + 1) if ks is None else ks
        return _sigmoid(Z @ self.E[ks].T)

    def p_within(self, X: np.ndarray, delta: int) -> np.ndarray:
        h = self.hazard(X, np.arange(1, min(delta, self.max_k) + 1))
        return 1.0 - np.prod(1.0 - h, axis=1)

    def nll(self, X, k, d) -> float:
        h = self.hazard(X)
        n = len(k); idx = np.arange(self.max_k)[None, :]
        surv = (idx < (k - 1)[:, None])
        ev = h[np.arange(n), k - 1]
        ll = np.sum(np.log(1 - h + 1e-9) * surv, 1) + d * np.log(ev + 1e-9) + (1 - d) * np.log(1 - ev + 1e-9)
        return float(-ll.mean())

    def fit(self, X: np.ndarray, k: np.ndarray, d: np.ndarray, steps: int = 300, batch: int = 256, seed: int = 0):
        """Maximise the censored log-likelihood. k in 1..max_k is the event/censoring offset, d the event flag."""
        r = np.random.default_rng(seed)
        k = np.clip(k, 1, self.max_k).astype(int); d = d.astype(float)
        hist = []
        for _ in range(steps):
            b = r.choice(len(X), min(batch, len(X)), replace=False)
            g = self._grad(X[b], k[b], d[b]); self._adam(g)
            hist.append(self.nll(X[b], k[b], d[b]))
        return hist

    def _grad(self, X, k, d):
        n = len(X); H, Z = self._embed(X)
        ks = np.arange(1, self.max_k + 1)
        h = _sigmoid(Z @ self.E[ks].T)                    # n x K
        idx = np.arange(self.max_k)[None, :]
        # target y_{uk}: 0 for k < k_u, d_u at k_u; mask beyond k_u
        mask = (idx <= (k - 1)[:, None]).astype(float)
        y = np.zeros_like(h); y[np.arange(n), k - 1] = d
        G = (h - y) * mask / n                            # dL/dlogit
        gE = np.zeros_like(self.E); gE[ks] = G.T @ Z
        gZ = G @ self.E[ks]
        gW2 = H.T @ gZ; gb2 = gZ.sum(0)
        gH = gZ @ self.W2.T * (H > 0)
        return dict(W1=X.T @ gH, b1=gH.sum(0), W2=gW2, b2=gb2, E=gE)

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for name, gr in g.items():
            self._m[name] = b1 * self._m[name] + (1 - b1) * gr
            self._v[name] = b2 * self._v[name] + (1 - b2) * gr * gr
            mh = self._m[name] / (1 - b1 ** self.t); vh = self._v[name] / (1 - b2 ** self.t)
            setattr(self, name, getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
