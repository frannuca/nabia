"""nabia_python_bridge: Python bindings of the nabia C++ numerical library.

Submodules mirror the library:

- ``stats``: descriptive statistics, normal distribution.
- ``signal``: EWMA, Butterworth band-pass, moving averages.
- ``forecast``: Kalman filter, Gaussian process (kernels), exponential smoothing.
- ``optimization``: L-BFGS, augmented Lagrangian, SQP, Nelder-Mead, differential
  evolution, with Python objectives.
- ``risk``: volatility, VaR, expected shortfall, risk budgeting.
- ``strategy``: risk budgeting portfolio backtests, performance metrics.
- ``regression``: elastic net and the factor regression engine.
- ``frames``: pandas interface to the backtest and the regression engine.

Arrays are numpy float64. Parameter structs take their fields as keyword
arguments, e.g. ``signal.EWMAParams(alpha=0.2)``. Errors raise ValueError
(invalid input) or RuntimeError. Long computations release the GIL.
"""

from __future__ import annotations

from . import forecast, optimization, regression, risk, signal, stats, strategy

__all__ = ["forecast", "frames", "optimization", "regression", "risk", "signal", "stats", "strategy"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # pandas is optional: import the frames module on first use.
    if name == "frames":
        import importlib

        return importlib.import_module(f"{__name__}.frames")
    raise AttributeError(f"module 'nabia_python_bridge' has no attribute {name!r}")
