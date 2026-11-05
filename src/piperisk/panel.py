from __future__ import annotations

import numpy as np
import pandas as pd

CELL = 0.02  # degrees (about 1.4 km east-west, 2.2 km north-south at Calgary)


def load_breaks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["lon", "lat"])
    df["year"] = df["date"].dt.year
    df["cx"] = np.floor(df["lon"] / CELL).astype(int)
    df["cy"] = np.floor(df["lat"] / CELL).astype(int)
    return df


def count_panel(df: pd.DataFrame, y0: int, y1: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Counts[cell, year] for cells that ever had a break; years y0..y1 inclusive."""
    g = df[(df.year >= y0) & (df.year <= y1)].groupby(["cx", "cy", "year"]).size()
    cells = sorted({(a, b) for a, b, _ in g.index})
    idx = {c: i for i, c in enumerate(cells)}
    M = np.zeros((len(cells), y1 - y0 + 1))
    for (a, b, y), n in g.items():
        M[idx[(a, b)], y - y0] = n
    return M, cells


def neighbour_sum(M: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    """Sum over the 8 neighbouring cells, same year."""
    idx = {c: i for i, c in enumerate(cells)}
    N = np.zeros_like(M)
    for (a, b), i in idx.items():
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if da == 0 and db == 0:
                    continue
                j = idx.get((a + da, b + db))
                if j is not None:
                    N[i] += M[j]
    return N


def features(M: np.ndarray, N: np.ndarray, t: int) -> np.ndarray:
    """Features for predicting year column t, using only columns < t."""
    f1 = M[:, t - 1]
    f2 = M[:, t - 2]
    f3 = M[:, t - 3]
    s5 = M[:, max(0, t - 5) : t].sum(1)
    cum = M[:, :t].sum(1)
    nb1 = N[:, t - 1]
    nb5 = N[:, max(0, t - 5) : t].sum(1)
    return np.column_stack([f1, f2, f3, s5, cum, nb1, nb5, np.full(len(M), t)])
