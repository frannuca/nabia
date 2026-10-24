// The nabia_python_bridge._core extension module: Python bindings of the whole
// nabia C++ library, one submodule per library area.

#include "bridge.hpp"

PYBIND11_MODULE(_core, m) {
    namespace nb = nabia_bridge;
    m.doc() = "Python bindings of the nabia C++ library";

    // Order matters: later areas use types registered by earlier ones
    // (BoxConstraints in optimization, Forecast in forecast, ...).
    auto stats = m.def_submodule("stats", "Descriptive statistics and the normal distribution");
    nb::bind_stats(stats);
    auto signal = m.def_submodule("signal", "Causal filters and smoothing");
    nb::bind_signal(signal);
    auto forecast = m.def_submodule("forecast", "Forecasters: Kalman filter, Gaussian process, exponential smoothing");
    nb::bind_forecast(forecast);
    auto optimization = m.def_submodule("optimization", "Gradient-based and derivative-free optimizers");
    nb::bind_optimization(optimization);
    auto risk = m.def_submodule("risk", "Risk measures and risk budgeting");
    nb::bind_risk(risk);
    auto strategy = m.def_submodule("strategy", "Risk budgeting portfolio backtests and performance metrics");
    nb::bind_strategy(strategy);
    auto regression = m.def_submodule("regression", "Elastic net and the factor regression engine");
    nb::bind_regression(regression);
}
