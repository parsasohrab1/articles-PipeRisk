"""Calgary water-main breaks (open data, 37 558 breaks 1956-2026): calibrated cell-level risk.

usage: python research/run_study.py <calgary_breaks.csv with columns date,type,status,lon,lat>
Panel: 0.02-degree cells with at least one break 1992-2025, years 1992-2025 (2026 is partial, excluded).
Target for year t: at least one break in the cell. Train t<=2012, calibrate 2013-2016, test 2017-2025.
The data contain only break locations and dates: no pipe age, material or diameter.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from piperisk.calib import ece, fit_isotonic, top_share
from piperisk.panel import count_panel, features, load_breaks, neighbour_sum

warnings.filterwarnings("ignore")
Y0, Y1 = 1992, 2025


def stack(M, N, years):
    X = np.vstack([features(M, N, t - Y0) for t in years])
    C = np.concatenate([M[:, t - Y0] for t in years])
    return X, C


def metrics(c, p):
    y = (c > 0).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "top5pct_share_of_breaks": top_share(c, p, 0.05),
        "top10pct_share_of_breaks": top_share(c, p, 0.10),
        "mean_pred": float(p.mean()),
        "event_rate": float(y.mean()),
    }


def main(path: str):
    df = load_breaks(path)
    M, cells = count_panel(df, Y0, Y1)
    N = neighbour_sum(M, cells)
    tr_y, ca_y, te_y = range(1997, 2013), range(2013, 2017), range(2017, 2026)
    Xtr, Ctr = stack(M, N, tr_y)
    Xca, Cca = stack(M, N, ca_y)
    Xte, Cte = stack(M, N, te_y)
    ytr = (Ctr > 0).astype(int)
    yca = (Cca > 0).astype(int)
    sc = StandardScaler().fit(np.log1p(Xtr[:, :-1]))
    f = lambda X: sc.transform(np.log1p(X[:, :-1]))
    models = {
        "logistic": LogisticRegression(max_iter=2000).fit(f(Xtr), ytr),
        "boosting": HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
            Xtr[:, :-1], ytr
        ),
    }
    pred = {
        "logistic": lambda X: models["logistic"].predict_proba(f(X))[:, 1],
        "boosting": lambda X: models["boosting"].predict_proba(X[:, :-1])[:, 1],
    }
    out = {
        "note": "real Calgary open data; cells with >=1 break 1992-2025; chronological split",
        "n_cells": len(cells),
        "n_breaks_used": int(M.sum()),
        "event_rate_by_period": {
            "train": float((Ctr > 0).mean()),
            "cal": float(yca.mean()),
            "test": float((Cte > 0).mean()),
        },
        "breaks_per_year": {str(Y0 + i): int(M[:, i].sum()) for i in range(M.shape[1])},
        "baselines": {},
        "models": {},
    }
    base = {"last_year_count": Xte[:, 0], "last_5y_count": Xte[:, 3], "cumulative_count": Xte[:, 4]}
    for k, s in base.items():
        out["baselines"][k] = {
            "auc": float(roc_auc_score((Cte > 0).astype(int), s)),
            "top5pct_share_of_breaks": top_share(Cte, s, 0.05),
            "top10pct_share_of_breaks": top_share(Cte, s, 0.10),
        }
    for name, fn in pred.items():
        pte = fn(Xte)
        iso = fit_isotonic(fn(Xca), yca)
        out["models"][name] = {
            "uncalibrated": metrics(Cte, pte),
            "isotonic_on_2013_2016": metrics(Cte, iso.predict(pte)),
        }
        # yearly reliability: mean predicted vs observed event rate per test year
        n = len(cells)
        per = {}
        for i, t in enumerate(te_y):
            sl = slice(i * n, (i + 1) * n)
            per[str(t)] = {"pred": float(pte[sl].mean()), "obs": float((Cte[sl] > 0).mean())}
        out["models"][name]["by_year"] = per
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))
    print(out["n_cells"], out["event_rate_by_period"])
    for k, v in out["baselines"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    for m, v in out["models"].items():
        for k in ("uncalibrated", "isotonic_on_2013_2016"):
            print(m, k, {a: round(b, 3) for a, b in v[k].items()})


if __name__ == "__main__":
    main(sys.argv[1])
