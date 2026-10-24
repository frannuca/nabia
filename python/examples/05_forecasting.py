"""Forecasting real series: Kalman filters, exponential smoothing, Gaussian processes.

- S&P 500 monthly realized volatility (noisy, mean-reverting): local level vs a
  custom mean-reverting AR(1)-plus-noise state-space model vs exponential smoothing.
- Gold (log price, monthly): local linear trend, with filtered/smoothed level and slope.
- Natural gas (log price, monthly): seasonality, with Gaussian process kernels built
  from squared-exponential, Matérn and periodic pieces.
Each model is scored on a holdout (last 12 or 24 months): RMSE and the coverage
of its 95% prediction intervals.

Covers: forecast.Forecast (mean, variance, horizon, stddev, lower, upper),
SimpleExponentialSmoothing(+Params), KalmanFilter(+Params: local_level,
local_linear_trend, custom), filtered_states, filtered_covariances, innovations,
innovation_variances, log_likelihood, smoothed_states, fitted_values,
GaussianProcess(+Params), predict, kernel, noise_variance, log_marginal_likelihood,
kernels (SquaredExponential, Matern52, Periodic, Sum, Product, k1 + k2, k1 * k2,
__call__, matrix, num_hyperparameters, log_hyperparameters, set_log_hyperparameters,
gradient, clone).

Run from the python/ folder (or in PyCharm, with python/.venv as interpreter):
    .venv/bin/python examples/05_forecasting.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _plots
from _data import header, load_prices
from nabia_python_bridge import forecast as fc


def holdout(model, y: np.ndarray, h: int) -> tuple[float, float, fc.Forecast]:
    """Fits on all but the last h points; RMSE and 95% coverage on them."""
    model.fit(y[:-h])
    f = model.forecast(h)
    actual = y[-h:]
    covered = np.mean((actual >= f.lower(0.95)) & (actual <= f.upper(0.95)))
    return float(np.sqrt(np.mean((actual - f.mean) ** 2))), float(covered), f


def ar1_plus_noise(phi: float, mean: float, state_var: float, obs_var: float) -> fc.KalmanFilterParams:
    """x[t] = mean + phi (x[t-1] - mean) + w as a 2-state model [x - mean, mean], y = x + v."""
    return fc.KalmanFilterParams(
        transition=np.array([[phi, 0.0], [0.0, 1.0]]),
        observation=np.array([[1.0, 1.0]]),
        process_noise=np.diag([state_var, 0.0]),
        observation_noise=obs_var,
        initial_state=np.array([0.0, mean]),
        initial_covariance=np.diag([state_var / (1 - phi ** 2), 1e-4]),
    )


def band(ax, x, f: fc.Forecast, color: str, label: str) -> None:
    """Forecast mean with its 95% interval."""
    ax.plot(x, f.mean, color=color, lw=1.4, label=label)
    ax.fill_between(x, f.lower(0.95), f.upper(0.95), color=color, alpha=0.15)


def plot_volatility(dates, vol: np.ndarray, holdouts: dict[str, fc.Forecast], h: int,
                    kf: fc.KalmanFilter, z: np.ndarray) -> None:
    fig, axes = _plots.figure(2, 1, "S&P 500 monthly realized volatility", size=(11, 7.5), height_ratios=[2, 1])
    ax = axes[0]
    ax.plot(dates, vol, color="0.3", lw=0.9, label="realized volatility")
    ax.plot(dates, kf.smoothed_states().sum(axis=0), color="C3", lw=1, alpha=0.7,
            label="AR(1) + noise: smoothed state")
    for k, (name, f) in enumerate(holdouts.items()):
        if name == "AR(1) + noise":
            band(ax, dates[-h:], f, f"C{k}", f"{name}: 12-month holdout forecast, 95% interval")
        else:
            ax.plot(dates[-h:], f.mean, color=f"C{k}", lw=1.4, label=f"{name}: 12-month holdout forecast")
    ax.axvline(dates[-h], color="k", ls=":", lw=1)
    _plots.percent(ax)
    ax.set(title="holdout forecasts (fit on data before the dotted line)")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.bar(dates, z, width=20, color=np.where(np.abs(z) > 1.96, "C3", "C0"))
    ax.axhspan(-1.96, 1.96, color="0.5", alpha=0.12)
    ax.set(title="AR(1) + noise: standardized innovations (red: outside +-1.96)")
    _plots.save(fig, "volatility")


def plot_gold(dates, gold: np.ndarray, trend: fc.KalmanFilter, holdout_fc: fc.Forecast, ahead: fc.Forecast) -> None:
    fig, axes = _plots.figure(2, 1, "Gold: local linear trend", size=(11, 7.5), sharex=True, height_ratios=[2, 1])
    level, slope = trend.smoothed_states()
    filtered_slope = trend.filtered_states()[1]
    ax = axes[0]
    ax.plot(dates, np.exp(gold), color="0.3", lw=0.9, label="gold (USD/oz)")
    ax.plot(dates, np.exp(level), color="C0", label="smoothed level")
    future = dates[-1] + pd.offsets.MonthEnd(1) * np.arange(1, ahead.horizon + 1)
    ax.plot(future, np.exp(ahead.mean), color="C1", label="12-month forecast")
    ax.fill_between(future, np.exp(ahead.lower(0.95)), np.exp(ahead.upper(0.95)), color="C1", alpha=0.2)
    h = holdout_fc.horizon
    ax.plot(dates[-h:], np.exp(holdout_fc.mean), color="C2", ls="--", label="24-month holdout forecast")
    ax.fill_between(dates[-h:], np.exp(holdout_fc.lower(0.95)), np.exp(holdout_fc.upper(0.95)), color="C2", alpha=0.12)
    ax.set_yscale("log")
    ax.set(title="price, smoothed level and forecasts with 95% intervals (log scale)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(dates, slope * 12, label="smoothed slope")
    ax.plot(dates, filtered_slope * 12, alpha=0.7, label="filtered slope (real time)")
    ax.axhline(0, color="k", lw=0.6)
    _plots.percent(ax)
    ax.set(title="trend slope, annualized")
    ax.legend(fontsize=8)
    _plots.save(fig, "gold_trend")


def plot_gas(dates, gas: np.ndarray, holdouts: dict[str, fc.Forecast], kernels: dict[str, fc.IKernel],
             gp: fc.GaussianProcess) -> None:
    fig, axes = _plots.figure(2, 2, "Natural gas: Gaussian processes", size=(13, 8))
    ax = axes[0, 0]
    h = 12
    ax.plot(dates[-60:], np.exp(gas[-60:]), "k.-", lw=0.8, ms=3, label="natural gas (USD/MMBtu)")
    for k, (name, f) in enumerate(holdouts.items()):
        ax.plot(dates[-h:], np.exp(f.mean), color=f"C{k}", label=name)
    best = min(holdouts, key=lambda n: np.mean((holdouts[n].mean - gas[-h:]) ** 2))
    f = holdouts[best]
    ax.fill_between(dates[-h:], np.exp(f.lower(0.95)), np.exp(f.upper(0.95)), color="0.5", alpha=0.2,
                    label=f"95% interval ({best})")
    ax.set(title="12-month holdout forecasts (last 5 years shown)")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(dates, np.exp(gas), color="0.3", lw=0.8, label="data")
    ax.plot(dates, np.exp(gp.fitted_values()), color="C0", lw=1, label="GP posterior mean (SE + periodic)")
    ahead = gp.forecast(12)
    future = dates[-1] + pd.offsets.MonthEnd(1) * np.arange(1, 13)
    ax.plot(future, np.exp(ahead.mean), color="C1", label="12-month forecast")
    ax.fill_between(future, np.exp(ahead.lower(0.95)), np.exp(ahead.upper(0.95)), color="C1", alpha=0.2)
    ax.set_yscale("log")
    ax.set(title="fit on everything and forecast (log scale)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    lag = np.linspace(0, 48, 400)
    for name, k in kernels.items():
        values = np.array([k(0.0, t) for t in lag])
        ax.plot(lag, values / values[0], label=name)
    ax.set(xlabel="lag (months)", ylabel="k(0, lag) / k(0, 0)", title="kernels as correlation vs lag (before fitting)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    t = np.arange(len(gas), dtype=float)
    ax.imshow(gp.kernel().matrix(t[-48:], t[-48:]), cmap="viridis")
    ax.grid(False)
    ax.set(title="fitted kernel: Gram matrix of the last 48 months", xlabel="month", ylabel="month")
    _plots.save(fig, "gas_gp")


def main() -> None:
    prices = load_prices(["^GSPC", "GC=F", "NG=F"], start="2010-01-01")
    daily = prices.pct_change().dropna()
    monthly = prices.resample("ME").last()

    # ------------------------------------------------------------------------------
    header("1. S&P 500 monthly realized volatility")
    vol = (daily["^GSPC"].groupby(daily.index.to_period("M")).std() * np.sqrt(252)).to_numpy()
    print(f"{len(vol)} months; mean {vol.mean():.1%}, last {vol[-1]:.1%}")
    h = 12
    var_y = np.var(vol)
    models = {
        "local level": fc.KalmanFilter(fc.KalmanFilterParams.local_level(0.1 * var_y, 0.5 * var_y,
                                                                         initial_level=vol[0])),
        "AR(1) + noise": fc.KalmanFilter(ar1_plus_noise(0.8, vol.mean(), 0.4 * var_y, 0.3 * var_y)),
        "SES(0.4)": fc.SimpleExponentialSmoothing(fc.SimpleExponentialSmoothingParams(alpha=0.4)),
    }
    vol_holdouts = {}
    for name, model in models.items():
        rmse, cover, f = holdout(model, vol, h)
        vol_holdouts[name] = f
        extra = f", log-lik {model.log_likelihood():.1f}" if isinstance(model, fc.KalmanFilter) else \
            f", SSE {model.sse:.3f} (alpha {model.alpha})"
        print(f"  {name:<14} 12-month RMSE {rmse:.2%}, 95% coverage {cover:.0%}, 12m-ahead {f.mean[-1]:.1%} "
              f"+- {f.stddev()[-1]:.1%}{extra}")
    print("  (the AR(1) model reverts to the long-run mean; local level and SES stay flat at the last level)")

    kf = models["AR(1) + noise"]
    kf.fit(vol)
    print(f"  fitted on all {len(vol)} months: state dim {kf.state_dim}, is_fitted {kf.is_fitted}, "
          f"transition phi {kf.parameters.transition[0, 0]}")
    states, smooth = kf.filtered_states(), kf.smoothed_states()
    print(f"  filtered states {states.shape}, smoothed {smooth.shape}, covariances {len(kf.filtered_covariances())} "
          f"x {kf.filtered_covariances()[0].shape}")
    z = kf.innovations() / np.sqrt(kf.innovation_variances())
    print(f"  standardized innovations: mean {np.nanmean(z):+.2f}, std {np.nanstd(z):.2f} (well specified: 0, 1)")
    vol_dates = daily["^GSPC"].groupby(daily.index.to_period("M")).std().index.to_timestamp("M")
    plot_volatility(vol_dates, vol, vol_holdouts, h, kf, z)
    gappy = vol.copy()
    gappy[-30:-24] = np.nan                                   # six missing months
    kf.fit(gappy)
    print(f"  with 6 missing months: log-lik {kf.log_likelihood():.1f}; the filter carries the state through "
          f"(one-step prediction at the gap {kf.fitted_values()[-27]:.1%})")

    # ------------------------------------------------------------------------------
    header("2. Gold: local linear trend (level + slope)")
    gold = np.log(monthly["GC=F"].to_numpy())
    trend = fc.KalmanFilter(fc.KalmanFilterParams.local_linear_trend(level_variance=2e-4, slope_variance=1e-5,
                                                                     observation_variance=5e-4))
    rmse, cover, f = holdout(trend, gold, 24)
    gold_holdout = f
    print(f"24-month holdout: RMSE {rmse:.3f} log points, 95% coverage {cover:.0%}")
    trend.fit(gold)
    level, slope = trend.smoothed_states()
    f = trend.forecast(12)
    print(f"current smoothed slope {slope[-1] * 12:+.1%}/yr (filtered {trend.filtered_states()[1, -1] * 12:+.1%}/yr); "
          f"slope a year ago {slope[-13] * 12:+.1%}/yr")
    print(f"12-month forecast: {np.exp(f.mean[-1]):.0f} USD/oz, 95% interval "
          f"[{np.exp(f.lower(0.95)[-1]):.0f}, {np.exp(f.upper(0.95)[-1]):.0f}] (last {np.exp(gold[-1]):.0f}); "
          f"horizon {f.horizon}, variance {f.variance[-1]:.4f}")
    plot_gold(monthly.index, gold, trend, gold_holdout, f)

    # ------------------------------------------------------------------------------
    header("3. Kernels")
    se = fc.SquaredExponentialKernel(variance=1.0, length_scale=24.0)
    per = fc.PeriodicKernel(variance=0.5, length_scale=1.0, period=12.0)
    mat = fc.Matern52Kernel(variance=0.2, length_scale=6.0)
    k = se + per * fc.SquaredExponentialKernel(1.0, 60.0)    # trend + slowly changing seasonality
    print(f"{type(k).__name__} (an IKernel: {isinstance(k, fc.IKernel)}) with {k.num_hyperparameters} hyperparameters; "
          f"k(0, 12) = {k(0.0, 12.0):.3f}, k(0, 6) = {k(0.0, 6.0):.3f} (a year apart is closer than 6 months)")
    print(f"log hyperparameters {np.round(k.log_hyperparameters(), 3)}; dk/dlog theta at (0, 6): "
          f"{np.round(k.gradient(0.0, 6.0), 3)}")
    t = np.arange(4.0)
    print(f"Gram matrix of a Matérn 5/2 on t = 0..3:\n{np.round(mat.matrix(t, t), 3)}")
    copy = mat.clone()
    copy.set_log_hyperparameters(np.log([0.5, 3.0]))
    print(f"clone is independent: original length {mat.length_scale}, clone {copy.length_scale:.1f}, "
          f"variance {copy.variance:.1f}; explicit constructors: {type(fc.SumKernel(se, mat)).__name__}, "
          f"{type(fc.ProductKernel(se, per)).__name__}; period {per.period}")

    # ------------------------------------------------------------------------------
    header("4. Natural gas: Gaussian processes with seasonal kernels (12-month holdout)")
    gas = np.log(monthly["NG=F"].to_numpy())
    candidates = {
        "SE only": fc.SquaredExponentialKernel(1.0, 12.0),
        "Matérn 5/2": fc.Matern52Kernel(1.0, 12.0),
        "SE + periodic": fc.SquaredExponentialKernel(1.0, 12.0) + fc.PeriodicKernel(0.1, 1.0, 12.0),
        "SE + periodic x SE": fc.SquaredExponentialKernel(1.0, 12.0)
                              + fc.PeriodicKernel(0.1, 1.0, 12.0) * fc.SquaredExponentialKernel(1.0, 48.0),
    }
    gas_holdouts = {}
    for name, kernel in candidates.items():
        params = fc.GaussianProcessParams(kernel=kernel, noise_variance=0.01, normalize=True,
                                          optimize_hyperparameters=True, optimize_noise=True,
                                          min_noise_variance=1e-6, max_iter=100)
        gp = fc.GaussianProcess(params)
        rmse, cover, f = holdout(gp, gas, 12)
        gas_holdouts[name] = f
        print(f"  {name:<20} RMSE {rmse:.3f}, 95% coverage {cover:.0%}, log marginal likelihood "
              f"{gp.log_marginal_likelihood():7.1f}, noise {gp.noise_variance():.2e}")
    print("  (front-month gas futures carry only a small seasonal swing: the likelihood, not the kernel's name,\n"
          "   decides whether a periodic component is worth its hyperparameters)")
    gp = fc.GaussianProcess(fc.GaussianProcessParams(kernel=candidates["SE + periodic"], noise_variance=0.01,
                                                     max_iter=100))
    gp.fit(gas)
    fitted = gp.kernel()
    print(f"\nfitted kernel ({type(fitted).__name__}): exp(log theta) = {np.round(np.exp(fitted.log_hyperparameters()), 3)}"
          f"; the input kernel is unchanged: {np.round(np.exp(gp.parameters.kernel.log_hyperparameters()), 3)}")
    mid = gp.predict(np.array([len(gas) - 6.5]))             # between two months, inside the data
    print(f"interpolation at t = T - 6.5: {np.exp(mid.mean[0]):.2f} USD +- {mid.stddev()[0]:.3f} log points; "
          f"in-sample fit RMSE {np.sqrt(np.mean((gp.fitted_values() - gas) ** 2)):.3f}")
    f = gp.forecast(12)
    months = np.exp(f.mean)
    print(f"next 12 months: low {months.min():.2f}, high {months.max():.2f} USD/MMBtu "
          f"(seasonal swing {months.max() / months.min() - 1:.0%})")
    plot_gas(monthly.index, gas, gas_holdouts, {"squared exponential (24)": se, "periodic (12)": per,
                                                "Matérn 5/2 (6)": mat, "SE + periodic x SE": k}, gp)
    _plots.show()


if __name__ == "__main__":
    main()
