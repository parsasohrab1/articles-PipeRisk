import numpy as np
import pandas as pd

from piperisk.calib import ece, fit_isotonic, top_share
from piperisk.panel import count_panel, features, neighbour_sum


def _df():
    rows = [
        (2000, 0.0, 0.0),
        (2000, 0.001, 0.001),
        (2001, 0.0, 0.0),
        (2001, 0.03, 0.0),
        (2002, 0.03, 0.0),
    ]
    d = pd.DataFrame(rows, columns=["year", "lon", "lat"])
    d["cx"] = np.floor(d.lon / 0.02).astype(int)
    d["cy"] = np.floor(d.lat / 0.02).astype(int)
    return d


def test_panel_counts_and_features_use_only_past():
    M, cells = count_panel(_df(), 2000, 2002)
    assert M.sum() == 5 and M.shape == (2, 3)
    N = neighbour_sum(M, cells)
    X = features(np.hstack([M, M]), np.hstack([N, N]), 3)
    # column 3 is a copy of column 0; changing M[:,3] must not change the features of year 3
    M2 = np.hstack([M, M * 5])
    X2 = features(M2, np.hstack([N, N * 5]), 3)
    assert np.array_equal(X, X2)


def test_ece_and_isotonic_and_top_share():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.4, 4000)
    y = (rng.uniform(size=4000) < p * 2).astype(int)  # model under-predicts by a factor 2
    iso = fit_isotonic(p[:2000], y[:2000])
    assert ece(y[2000:], iso.predict(p[2000:])) < ece(y[2000:], p[2000:])
    assert top_share(np.array([0, 0, 5, 1]), np.array([0.1, 0.2, 0.9, 0.5]), 0.25) == 5 / 6
