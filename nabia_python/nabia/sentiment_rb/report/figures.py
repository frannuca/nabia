"""Figures: portfolio.png, budgets.png, model.png, and the IC sweep chart."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pyplot(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7})
    return matplotlib, plt


def plots(cfg, out, bench, overlay, sched, s, b0, assets, ics, selected, signal_start, show):
    """portfolio.png (PnL, drawdown, Sharpe, yearly), budgets.png (b, s, w, RC), model.png (IS vs OOS, selection)."""
    matplotlib, plt = _pyplot(show)
    B = cfg["budgets"]
    runs = {f"{k} (Sharpe {v.metrics['sharpe_ratio']:.2f})": v
            for k, v in [("ERC (b0)", bench), (f"sentiment overlay, {B['mode'].replace('_', '-')}", overlay)]}
    perf = {k: v.performance for k, v in runs.items()}
    erc, ovl = perf.values()
    oos = lambda ax: ax.axvspan(erc.index[0], signal_start, color="grey", alpha=0.12, lw=0)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle("Sentiment overlay vs plain ERC (shaded: covariance warm-up, before the first out-of-sample signal)")
    for k, p in perf.items():
        ax[0, 0].plot(p.equity / p.equity.iloc[0] - 1, label=k)
        ax[1, 0].fill_between(p.index, -p.drawdown, alpha=0.4, label=k)
        r = p.net_returns
        ax[0, 1].plot(r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252), label=k)
    ax[0, 0].plot(ovl.equity / erc.equity - 1, "k--", lw=0.8, label="overlay vs ERC")
    yearly = pd.DataFrame({k: (1 + p.net_returns).groupby(p.index.year).prod() - 1 for k, p in perf.items()})
    yearly.plot.bar(ax=ax[1, 1], width=0.8)
    for a, title in zip(ax.flat, ["cumulative PnL (net of costs)", "rolling 1y Sharpe ratio", "drawdown", "return by year"]):
        a.set_title(title), a.legend()
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        oos(a)
    for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(), fig.savefig(out / "portfolio.png", dpi=120)

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.tab10(np.arange(len(assets)))
    ax[0].stackplot(sched.index, sched.T.values, labels=assets, colors=colors, alpha=0.85, step="post")
    for b, c in zip(np.cumsum(b0)[:-1], colors):
        ax[0].axhline(b, color="k", ls=":", lw=0.7)
    if B["mode"] == "long_short" and not B["modulate_long_short"]:
        ax[0].set_title("risk budgets b, fixed at b0 in long-short mode: sentiment sets the side of each position "
                        "(see the weights below); \"modulate_long_short\": true scales them by conviction")
    elif B["mode"] == "long_short":
        ax[0].set_title("risk budgets b, scaled by conviction (s for longs, 1 - s for shorts; dotted: base budgets b0)")
    else:
        ax[0].set_title("risk budgets b (stacked; dotted: base budgets b0)")
    for a, c in zip(assets, colors):
        ax[1].plot(s[a], color=c, lw=0.8, label=a)
    ax[1].axhline(0.5, color="k", lw=0.7), ax[1].set_ylim(0, 1), ax[1].set_title("normalized sentiment s (0.5 = neutral)")
    w, rc = overlay.target_weights, overlay.risk_contributions
    if B["mode"] == "long_short":  # neutral band: short below, long again above
        ax[1].axhspan(B["short_below"], B["long_above"], color="grey", alpha=0.15, label="neutral band")
        ax[1].axhline(B["short_below"], color="r", lw=0.8, ls="--", label="short below")
        for a, c in zip(assets, colors):
            ax[2].step(w.index, w[a], where="post", color=c, lw=1, label=a)
        ax[2].axhline(0, color="k", lw=0.7)
        ax[2].set_title("overlay weights (negative = short; each position still holds its risk budget)")
    else:
        ax[2].stackplot(w.index, w.T.values, colors=colors, alpha=0.85, step="post")
        ax[2].plot(rc.cumsum(axis=1).iloc[:, :-1], color="k", lw=0.6, ls="--")
        ax[2].set_title("overlay weights (stacked) and cumulative risk contributions (dashed, = budgets)")
    for a in ax:
        a.legend(loc="upper left", ncol=len(assets)) if a is not ax[2] else None
        a.margins(x=0)
    fig.tight_layout(), fig.savefig(out / "budgets.png", dpi=120)

    fig = plt.figure(figsize=(12, 8))
    g = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    ax0, ax1, ax2 = fig.add_subplot(g[0, 0]), fig.add_subplot(g[1, 0]), fig.add_subplot(g[:, 1])
    ics.groupby("asset")[["in_sample", "out_of_sample"]].mean().reindex(assets).plot.bar(ax=ax0, rot=0)
    ax0.set_title("elastic net IC: on its training rows vs the next unseen block (mean over refits)")
    ics.groupby("refit")[["in_sample", "out_of_sample"]].mean().plot(ax=ax1, marker=".")
    ax1.axhline(0, color="k", lw=0.7), ax1.set_title("IC per refit (mean over assets with a model)")
    top = selected.loc[selected.max(axis=1).sort_values(ascending=False).index[:25]]
    im = ax2.imshow(top.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(top.columns)), top.columns), ax2.set_yticks(range(len(top)), top.index)
    ax2.set_title("share of refits with a non-zero beta (top 25 features)"), ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.04)
    fig.tight_layout(), fig.savefig(out / "model.png", dpi=120)
    plt.show() if show else plt.close("all")


def plot_sweep(out, sweep, show):
    """sweep.png: Sharpe, excess return and realized IC against the synthetic news IC."""
    _, plt = _pyplot(show)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sweep[["erc_sharpe", "overlay_sharpe"]].plot(ax=ax[0], marker="o", title="Sharpe ratio")
    (sweep.excess_return * 100).plot(ax=ax[1], marker="o", color="k", title="overlay - ERC, annualized return (%)")
    sweep[["ic_S", "ic_llm_level"]].plot(ax=ax[2], marker="o", title="realized IC with the next 21-day return")
    ax[2].plot(sweep.index, sweep.index, "k:", lw=0.8, label="news tone IC (set)"), ax[2].legend()
    for a in ax:
        a.set_xlabel("synthetic news IC (corr of tone with forward return)"), a.axhline(0, color="k", lw=0.6)
    fig.tight_layout(), fig.savefig(out / "sweep.png", dpi=120)
    plt.show() if show else plt.close("all")
