"""Elastic net per asset (nabia regression engine: purged walk-forward CV, positivity, stability selection)
and the walk-forward sentiment score S and its normalization s in [0,1]."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from nabia_python_bridge import frames, regression, signal, stats


def fit_asset(cfg, X, y):
    M, cols = cfg["model"], list(X.columns)
    while cols:
        res = frames.regress(y, X[cols], min_factors=len(cols), max_factors=len(cols), train_fraction=1.0,
                             alphas=M["l1_ratios"], validation=regression.ValidationScheme.WalkForward,
                             validation_folds=M["cv_folds"], validation_gap=cfg["horizon"],
                             bootstrap_samples=M["stability"]["samples"], bootstrap_block_size=cfg["horizon"])
        beta, freq = res.coefficients, res.bootstrap.selection_frequency.reindex(cols).fillna(0)
        drop = set(beta[beta < 0].index if M["positive"] else []) | set(freq[freq <= M["stability"]["threshold"]].index)
        if not drop:
            return beta.reindex(X.columns).fillna(0.0)
        cols = [c for c in cols if c not in drop]
    return pd.Series(0.0, X.columns)  # nothing survives: S = 0, neutral


def walk_forward_scores(cfg, X_by_asset, y_by_asset, dates):
    """S (ex-intercept) and s in [0,1] per asset and day, with betas refit on the configured schedule."""
    M, N, h = cfg["model"], cfg["normalize"], cfg["horizon"]
    ends = pd.Series(dates, dates).resample(M["refit"]).last().values
    refits = [dates.get_loc(d) for d in ends if dates.get_loc(d) >= cfg["z_min"] + M["min_train"] + h]
    S, betas, ics = pd.DataFrame(np.nan, dates, list(X_by_asset)), {}, []
    s = S.copy()
    def fit_at(p, a):
        X, lo = X_by_asset[a], 0 if M["window"] == "expanding" else max(0, p - h - M["rolling_days"])
        Xw, yw = X.iloc[lo:p - h + 1], y_by_asset[a].iloc[lo:p - h + 1]
        rows = (Xw.notna().mean(axis=1) >= 0.5) & yw.notna()  # past the z-score warm-up; missing z = neutral 0
        assert not rows.any() or yw[rows].index.max() + pd.tseries.offsets.BDay(h) <= dates[p], \
            "purge: target overlaps the refit date"
        if rows.sum() < M["min_train"]:
            return pd.Series(0.0, X.columns), np.nan
        beta = fit_asset(cfg, Xw[rows].fillna(0.0), yw[rows])
        return beta, np.corrcoef(Xw[rows].fillna(0.0).values @ beta.values, yw[rows])[0, 1] if beta.any() else np.nan

    # nabia releases the GIL, so the assets' fits run in parallel threads; "threads": 1 fits them one after the
    # other (simpler to step through in a debugger), 0 uses one thread per core.
    threads = M.get("threads", 0)
    pool = ThreadPoolExecutor(max_workers=threads or None) if threads != 1 else None
    fits = (lambda p: pool.map(lambda a: fit_at(p, a), X_by_asset)) if pool else (lambda p: map(lambda a: fit_at(p, a), X_by_asset))
    for k, p in enumerate(refits):
        stop = refits[k + 1] if k + 1 < len(refits) else len(dates) - 1
        for a, (beta, is_ic) in zip(X_by_asset, fits(p)):
            X, betas[(dates[p], a)] = X_by_asset[a], beta
            cur = X.fillna(0.0).values @ beta.values  # S_t with the current betas
            oos = pd.Series(cur[p + 1:stop + 1]).corr(y_by_asset[a].iloc[p + 1:stop + 1].reset_index(drop=True))
            ics.append({"refit": dates[p], "asset": a, "in_sample": is_ic, "out_of_sample": oos if beta.any() else np.nan})
            for t in range(p + 1, stop + 1):
                hist = cur[max(0, t - N["lookback"]):t]
                S.iat[t, S.columns.get_loc(a)] = cur[t]
                s.iat[t, s.columns.get_loc(a)] = ((hist < cur[t]).mean() + 0.5 * (hist == cur[t]).mean()  # mid-rank: S = 0 always -> 0.5
                                                  if N["method"] == "percentile" else
                                                  float(stats.normal_cdf(np.array([cur[t] / (hist.std() or 1)]))[0]))
    live = S.dropna(how="all").index
    xs = (S.loc[live].rank(axis=1) - 1) / (S.shape[1] - 1)  # cross-sectional rank in [0,1]
    blend = N["alpha_ts"] * s.loc[live] + (1 - N["alpha_ts"]) * xs
    smooth = blend.apply(lambda c: signal.exponential_moving_average(c.values, 1 - 0.5 ** (1 / N["ewma_halflife"])))
    assert ((smooth >= 0) & (smooth <= 1)).all().all()
    return S.loc[live], smooth, betas, pd.DataFrame(ics)
