// Python bindings: the nabia._nabia extension module.
//
// Thin entry points over the C++ library. Dates cross the boundary as int64
// days since 1970-01-01 (numpy datetime64[D]); prices as a float64 (dates x
// assets) matrix; portfolio parameters as the JSON text understood by the
// nabia_rb_backtest application, so Python and the CLI accept exactly the same
// keys. The Python package (python/nabia) wraps these in pandas objects.

#include "config.hpp"
#include "csv_io.hpp"

#include <nabia/dataframe/DataFrame.h>
#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using DaysArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<Date> to_dates(const DaysArray& days) {
    std::vector<Date> out;
    out.reserve(static_cast<std::size_t>(days.size()));
    for (const std::int64_t d : std::span(days.data(), static_cast<std::size_t>(days.size()))) {
        out.push_back(Date{std::chrono::days{d}});
    }
    return out;
}

DaysArray to_days(const std::vector<Date>& dates) {
    DaysArray out(static_cast<py::ssize_t>(dates.size()));
    auto view = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < dates.size(); ++i) {
        view(static_cast<py::ssize_t>(i)) = dates[i].time_since_epoch().count();
    }
    return out;
}

/// Returns ready for the engine, with the configuration that produced them.
struct Prepared {
    DataFrame<Date> returns;
    nabia::app::BacktestConfig config;
    Index dropped_dates = 0;
};

/// The same steps as nabia_rb_backtest: parse the configuration, drop dates
/// where some asset has no price, apply start/end, compute simple returns and
/// resolve the budgets against the asset names.
Prepared prepare(const DaysArray& days, const std::vector<std::string>& assets, const MatrixXd& prices,
                 const std::string& config_json) {
    if (prices.rows() != days.size() || prices.cols() != static_cast<Index>(assets.size())) {
        throw std::invalid_argument("prices must be (dates x assets): got " + std::to_string(prices.rows()) + " x " +
                                    std::to_string(prices.cols()) + " for " + std::to_string(days.size()) +
                                    " dates and " + std::to_string(assets.size()) + " assets");
    }
    Prepared p;
    p.config = nabia::app::parse_config(config_json);
    DataFrame<Date> px(to_dates(days), assets, prices);
    const Index read = px.rows();
    px = px.drop_missing();
    p.dropped_dates = read - px.rows();
    if (px.rows() >= 2 && (p.config.start || p.config.end)) {
        px = px.between(p.config.start.value_or(px.time(0)), p.config.end.value_or(px.time(px.rows() - 1)));
    }
    if (px.rows() < 2) {
        throw std::invalid_argument("need at least 2 dates with a price for every asset");
    }
    for (Index t = 0; t < px.rows(); ++t) {
        for (Index j = 0; j < px.cols(); ++j) {
            if (!(px.values()(t, j) > 0.0)) {
                throw std::invalid_argument("prices must be positive for returns: " + px.columns()[static_cast<std::size_t>(j)] +
                                            " is " + std::to_string(px.values()(t, j)) + " on " +
                                            nabia::app::format_date(px.time(t)));
            }
        }
    }
    p.returns = px.pct_change();
    p.config.params.budgets = nabia::app::resolve_budgets(p.config, assets);
    return p;
}

py::dict metrics_dict(const PerformanceMetrics& m) {
    py::dict d;
    d["periods"] = m.periods;
    d["total_return"] = m.total_return;
    d["annualized_return"] = m.annualized_return;
    d["annualized_volatility"] = m.annualized_volatility;
    d["sharpe_ratio"] = m.sharpe_ratio;
    d["sortino_ratio"] = m.sortino_ratio;
    d["max_drawdown"] = m.max_drawdown;
    d["max_drawdown_duration"] = m.max_drawdown_duration;
    d["calmar_ratio"] = m.calmar_ratio;
    d["hit_rate"] = m.hit_rate;
    d["best_period"] = m.best_period;
    d["worst_period"] = m.worst_period;
    d["rebalances"] = m.rebalances;
    d["average_turnover"] = m.average_turnover;
    d["annualized_turnover"] = m.annualized_turnover;
    d["total_costs"] = m.total_costs;
    return d;
}

py::dict backtest(const DaysArray& days, const std::vector<std::string>& assets, const MatrixXd& prices,
                  const std::string& config_json, const std::optional<DaysArray>& schedule_dates,
                  const std::optional<MatrixXd>& schedule_budgets) {
    Prepared p = prepare(days, assets, prices, config_json);
    if (schedule_dates.has_value() != schedule_budgets.has_value()) {
        throw std::invalid_argument("schedule_dates and schedule_budgets must be given together");
    }
    std::optional<DataFrame<Date>> schedule;
    if (schedule_dates) {
        if (schedule_budgets->rows() != schedule_dates->size() || schedule_budgets->cols() != static_cast<Index>(assets.size())) {
            throw std::invalid_argument("schedule_budgets must be (schedule dates x assets)");
        }
        schedule.emplace(to_dates(*schedule_dates), assets, *schedule_budgets);
    }
    std::unique_ptr<PortfolioResult<Date>> r;
    {
        // Pure C++ from here on: let other Python threads run meanwhile.
        const py::gil_scoped_release release;
        const RiskBudgetingPortfolio engine(std::make_unique<RiskBudgetingPortfolioParams>(p.config.params));
        r = std::make_unique<PortfolioResult<Date>>(schedule ? engine.run(p.returns, *schedule) : engine.run(p.returns));
    }
    py::dict out;
    out["dates"] = to_days(r->weights.index());
    out["assets"] = r->weights.columns();
    out["weights"] = r->weights.values();
    out["rebalance_dates"] = to_days(r->target_weights.index());
    out["target_weights"] = r->target_weights.values();
    out["risk_contributions"] = r->risk_contributions.values();
    out["leverage"] = r->leverage.values();
    out["gross_returns"] = r->gross_returns.values();
    out["costs"] = r->costs.values();
    out["net_returns"] = r->net_returns.values();
    out["turnover"] = r->turnover.values();
    out["equity"] = r->equity.values();
    out["pnl"] = r->pnl.values();
    out["drawdown"] = r->drawdown.values();
    out["metrics"] = metrics_dict(r->metrics);
    out["all_converged"] = r->all_converged;
    out["threads_used"] = r->threads_used;
    out["dropped_dates"] = p.dropped_dates;
    return out;
}

py::dict latest_weights(const DaysArray& days, const std::vector<std::string>& assets, const MatrixXd& prices,
                        const std::string& config_json) {
    Prepared p = prepare(days, assets, prices, config_json);
    const Index window = p.config.params.covariance_window;
    if (p.returns.rows() < window) {
        throw std::invalid_argument("need at least covariance_window = " + std::to_string(window) +
                                    " returns, i.e. " + std::to_string(window + 1) + " price dates; got " +
                                    std::to_string(p.returns.rows()) + " returns");
    }
    // A backtest over the last window has a single rebalance, at the last date.
    const DataFrame<Date> recent = p.returns.tail(window);
    std::unique_ptr<PortfolioResult<Date>> r;
    {
        const py::gil_scoped_release release;
        auto params = std::make_unique<RiskBudgetingPortfolioParams>(p.config.params);
        params->rebalance_frequency = 1;
        r = std::make_unique<PortfolioResult<Date>>(RiskBudgetingPortfolio(std::move(params)).run(recent));
    }
    py::dict out;
    out["date"] = to_days({r->target_weights.time(0)});
    out["assets"] = r->target_weights.columns();
    out["weights"] = VectorXd(r->target_weights.values().row(0).transpose());
    out["risk_contributions"] = VectorXd(r->risk_contributions.values().row(0).transpose());
    out["leverage"] = r->leverage[0];
    out["converged"] = r->all_converged;
    out["dropped_dates"] = p.dropped_dates;
    return out;
}

} // namespace

PYBIND11_MODULE(_nabia, m) {
    m.doc() = "nabia C++ core: risk budgeting portfolio backtests";
    m.def("backtest", &backtest, py::arg("dates"), py::arg("assets"), py::arg("prices"), py::arg("config_json"),
          py::arg("schedule_dates") = py::none(), py::arg("schedule_budgets") = py::none(),
          "Walk-forward risk budgeting backtest.\n\n"
          "dates: int64 days since 1970-01-01, strictly increasing; assets: names; prices: (dates x assets)\n"
          "float64, NaN = missing (such dates are dropped); config_json: portfolio parameters as JSON.\n"
          "Optional schedule_dates (int64 days) and schedule_budgets (dates x assets): risk budgets decided\n"
          "on each date, used from the first rebalance at or after it.\n"
          "Returns a dict of numpy arrays (see nabia.backtest for the pandas interface).");
    m.def("latest_weights", &latest_weights, py::arg("dates"), py::arg("assets"), py::arg("prices"),
          py::arg("config_json"),
          "Risk budgeting weights to hold from the last date on, from the last covariance_window returns.");
}
