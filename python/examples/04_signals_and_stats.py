"""Statistics and trend signals on real prices.

Part 1 checks two stylized facts of daily returns with nabia.stats: fat tails
(more extreme days than the normal distribution predicts) and volatility
clustering (squared returns are autocorrelated, returns are not).
Part 2 extracts trends with the causal filters of nabia.signal and turns an
EWMA crossover into a long/flat trend-following rule.

Covers: stats.mean, variance, stddev, autocorrelation, normal_pdf, normal_cdf,
normal_quantile; signal.moving_average, exponential_moving_average, diff,
EWMA (process, reset, apply, parameters, alpha_from_span, alpha_from_halflife),
EWMAParams, ButterworthBandPass (process, reset, apply, parameters, max_order),
ButterworthBandPassParams.

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/04_signals_and_stats.py
"""

from __future__ import annotations

import numpy as np

import _plots
from _data import header, load_prices
from nabia_python_bridge import signal, stats, strategy


def stylized_facts(name: str, prices: np.ndarray) -> np.ndarray:
    log_returns = signal.diff(np.log(prices))            # first difference of log prices
    m, s = stats.mean(log_returns), stats.stddev(log_returns)
    print(f"\n{name}: {len(log_returns)} daily log returns")
    print(f"  mean {m * 252:.2%}/yr, stddev {s * np.sqrt(252):.2%}/yr "
          f"(population variance {stats.variance(log_returns, ddof=0):.2e} vs sample {stats.variance(log_returns):.2e})")
    z = (log_returns - m) / s
    for k in (2.0, 3.0, 4.0):
        expected = 2 * (1 - stats.normal_cdf(k))
        observed = np.mean(np.abs(z) > k)
        print(f"  |z| > {k:.0f}: observed {observed:.3%}, normal {expected:.3%} (x{observed / expected:.1f})")
    q = stats.normal_quantile(0.01)
    print(f"  1% normal quantile z = {q:.3f}: Gaussian 1-day 99% VaR {-(m + q * s):.2%}, "
          f"historical {-np.quantile(log_returns, 0.01):.2%}")
    dens = stats.normal_pdf(np.array([0.0, 1.0, 2.0]))   # vectorized
    print(f"  normal density at 0, 1, 2: {np.round(dens, 4)}")
    for lag in (1, 5, 21):
        print(f"  autocorrelation lag {lag:>2}: returns {stats.autocorrelation(log_returns, lag):+.3f}, "
              f"squared returns {stats.autocorrelation(log_returns ** 2, lag):+.3f}")
    return log_returns


def plot_stylized_facts(series: dict[str, np.ndarray]) -> None:
    """Return distribution against the normal, tail frequencies, and autocorrelations."""
    fig, axes = _plots.figure(1, 3, "Stylized facts of daily returns", size=(15, 4.2))
    grid = np.linspace(-8, 8, 400)
    for k, (name, r) in enumerate(series.items()):
        z = (r - stats.mean(r)) / stats.stddev(r)
        axes[0].hist(z, bins=120, range=(-8, 8), density=True, histtype="step", lw=1.2, color=f"C{k}", label=name)
        ks = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ratio = [np.mean(np.abs(z) > q) / (2 * (1 - stats.normal_cdf(q))) for q in ks]
        axes[1].bar(ks + (k - 0.5) * 0.35, ratio, 0.35, label=name, color=f"C{k}")
    axes[0].plot(grid, stats.normal_pdf(grid), "k--", lw=1, label="standard normal")
    axes[0].set_yscale("log")
    axes[0].set(ylim=(1e-5, 1), xlabel="standardized return", title="density (log scale): fat tails")
    axes[0].legend()
    axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="|z| greater than", ylabel="observed / normal frequency", title="tail frequency vs normal")
    axes[1].legend()
    lags = np.arange(1, 61)
    r = next(iter(series.values()))
    axes[2].bar(lags, [stats.autocorrelation(r, int(l)) for l in lags], color="0.6", label="returns")
    axes[2].plot(lags, [stats.autocorrelation(r ** 2, int(l)) for l in lags], "C3o-", ms=3, label="squared returns")
    band = 1.96 / np.sqrt(len(r))
    axes[2].axhspan(-band, band, color="C0", alpha=0.15, label="95% band (no autocorrelation)")
    axes[2].set(xlabel="lag (days)", title=f"{next(iter(series))}: volatility clustering")
    axes[2].legend()
    _plots.save(fig, "stylized_facts")


def plot_trend(dates, spx: np.ndarray, sma: np.ndarray, ema: np.ndarray, cycle: np.ndarray,
               crossovers: dict[str, np.ndarray], hold: np.ndarray) -> None:
    """Smoothers on the price, the band-pass cycle, and the crossover rules against buy & hold."""
    fig, axes = _plots.figure(3, 1, "Trend filters on the S&P 500", size=(11, 10), sharex=True)
    ax = axes[0]
    ax.plot(dates, spx, color="0.3", lw=0.8, label="S&P 500")
    ax.plot(dates[len(spx) - len(sma):], sma, label="200-day SMA")
    ax.plot(dates, ema, label="EMA, span 200")
    ax.set_yscale("log")
    ax.set(title="price and smoothers (log scale)")
    ax.legend()
    ax = axes[1]
    ax.plot(dates[300:], cycle[300:], color="C2", lw=0.9)
    ax.axhline(0, color="k", lw=0.6)
    _plots.percent(ax)
    ax.set(title="Butterworth band-pass of log price: 63-252 day cycles (first 300 days are warm-up)")
    ax = axes[2]
    for name, r in crossovers.items():
        ax.plot(dates[-len(r):], strategy.equity_curve(r, start=1.0), lw=1, label=f"EWMA {name}")
    ax.plot(dates[-len(hold):], strategy.equity_curve(hold, start=1.0), "k--", lw=1.2, label="buy & hold")
    ax.set_yscale("log")
    ax.set(title="long/flat crossover rules: equity (log scale)")
    ax.legend()
    _plots.save(fig, "trend")


def main() -> None:
    prices = load_prices(["^GSPC", "GC=F"], start="2005-01-01")
    spx, gold = prices["^GSPC"].to_numpy(), prices["GC=F"].to_numpy()

    header("1. Stylized facts of daily returns")
    facts = {"S&P 500": stylized_facts("S&P 500", spx), "Gold": stylized_facts("Gold", gold)}
    plot_stylized_facts(facts)

    # ------------------------------------------------------------------------------
    header("2. Smoothing and causal trend filters")
    sma = signal.moving_average(spx, 200)                       # len(x) - 199 points
    ema = signal.exponential_moving_average(spx, 2 / 201)       # same centre of mass
    print(f"S&P 500 last {spx[-1]:.0f}; 200-day SMA {sma[-1]:.0f}; EMA {ema[-1]:.0f} "
          f"(lengths {len(sma)} and {len(ema)} for {len(spx)} prices)")

    alpha = signal.EWMA.alpha_from_span(200)
    print(f"alpha for a 200-day span {alpha:.5f} (= 2/201: {np.isclose(alpha, 2 / 201)}), "
          f"for a 20-day half-life {signal.EWMA.alpha_from_halflife(20):.5f}")
    ewma = signal.EWMA(signal.EWMAParams(alpha=alpha, seed_with_first_sample=True))
    batch = ewma.apply(spx)                                      # whole series from a fresh state
    live = np.array([ewma.process(p) for p in spx])              # streaming, as in production
    print(f"EWMA batch == streaming: {np.allclose(batch, live)}; equals exponential_moving_average: "
          f"{np.allclose(batch, ema)}; alpha kept {ewma.parameters.alpha:.5f}")
    ewma.reset()
    print(f"after reset() the first output is the first sample again: {ewma.process(spx[0]) == spx[0]}")

    # Band-pass on log prices: keep cycles between ~3 months and ~1 year (sample rate 1/day).
    bp_params = signal.ButterworthBandPassParams(order=2, sample_rate=1.0, low_cutoff=1 / 252, high_cutoff=1 / 63)
    bp = signal.ButterworthBandPass(bp_params)
    cycle = bp.apply(np.log(spx))
    streamed = np.array([bp.process(x) for x in np.log(spx)])
    print(f"band-pass (order {bp.parameters.order}, max order {signal.ButterworthBandPass.max_order}) "
          f"keeps 63-252 day cycles: current cycle {cycle[-1]:+.2%} of price, "
          f"cycle std {cycle[300:].std():.2%}; streaming matches batch: {np.allclose(cycle, streamed)}")
    bp.reset()
    turning = np.flatnonzero(np.diff(np.sign(cycle[-260:])))
    print(f"cycle zero crossings in the last year: {len(turning)}")

    # ------------------------------------------------------------------------------
    header("3. EWMA crossover trend following (long when fast > slow, else flat)")
    returns = np.diff(spx) / spx[:-1]
    crossovers = {}
    for fast_span, slow_span in [(20, 100), (50, 200), (10, 50)]:
        fast = signal.EWMA(alpha=signal.EWMA.alpha_from_span(fast_span)).apply(spx)
        slow = signal.EWMA(alpha=signal.EWMA.alpha_from_span(slow_span)).apply(spx)
        position = (fast > slow).astype(float)[:-1]             # decided at the close, earns the next day
        strat = position * returns
        warm = slow_span                                          # skip the filters' warm-up
        m_strat = strategy.compute_performance(strat[warm:], 252)
        m_hold = strategy.compute_performance(returns[warm:], 252)
        trades = int(np.abs(np.diff(position[warm:])).sum())
        crossovers[f"{fast_span}/{slow_span}"] = strat[200:]                # common start for the plot
        print(f"  {fast_span:>3}/{slow_span:<3}: Sharpe {m_strat.sharpe_ratio:.2f} vs buy & hold "
              f"{m_hold.sharpe_ratio:.2f}; max drawdown {m_strat.max_drawdown:.1%} vs {m_hold.max_drawdown:.1%}; "
              f"invested {position[warm:].mean():.0%} of days, {trades} switches")
    plot_trend(prices.index, spx, sma, ema, cycle, crossovers, returns[200:])
    _plots.show()


if __name__ == "__main__":
    main()
