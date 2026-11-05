from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(c) / n * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def top_share(y_count: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Share of all breaks that fall in the top ``frac`` cells ranked by risk."""
    k = max(1, int(frac * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y_count[top].sum() / max(y_count.sum(), 1))


def fit_isotonic(p_cal: np.ndarray, y_cal: np.ndarray):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
