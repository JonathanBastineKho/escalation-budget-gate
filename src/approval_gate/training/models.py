"""Small regularized logistic heads stored as inspectable JSON, not pickle."""

from __future__ import annotations
import numpy as np


def fit(values, labels, l2=0.1, iterations=1200):
    x, y = np.asarray(values, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 2 or len(y) != len(x) or len(y) < 2:
        raise ValueError("at least two labeled examples are required")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("features must be finite; targets must be binary")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ w + b, -40, 40)))
        error = p - y
        w -= 0.05 * (x.T @ error / len(y) + l2 * w)
        b -= 0.05 * error.mean()
    return {
        "weights": w.tolist(),
        "bias": float(b),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "n": len(y),
        "positives": int(y.sum()),
        "l2": l2,
    }
