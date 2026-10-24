#include "bridge.hpp"

#include <nabia/strategy/PerformanceMetrics.h>
#include <nabia/strategy/RiskBudgetingPortfolio.h>

#include <optional>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace nabia_bridge {

namespace {

namespace sd = nabia::strategy::detail;

/// Budget changes from (rows, budgets): row k of `budgets` applies from returns row rows[k] on.
std::vector<sd::BudgetChange> schedule(const std::optional<std::vector<Index>>& rows,
                                       const std::optional<MatrixXd>& budgets, Index assets) {
    if (rows.has_value() != budgets.has_value()) {
        throw std::invalid_argument("schedule_rows and schedule_budgets must be given together");
    }
    std::vector<sd::BudgetChange> out;
    if (!rows) {
        return out;
    }
    if (budgets->rows() != static_cast<Index>(rows->size()) || budgets->cols() != assets) {
        throw std::invalid_argument("schedule_budgets must be (len(schedule_rows) x assets)");
    }
    for (std::size_t k = 0; k < rows->size(); ++k) {
        if (k > 0 && (*rows)[k] < (*rows)[k - 1]) {
            throw std::invalid_argument("schedule_rows must be sorted");
        }
        out.push_back({(*rows)[k], budgets->row(static_cast<Index>(k)).transpose()});
    }
    return out;
}

} // namespace

void bind_strategy(py::module_& m) {
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readonly("periods", &PerformanceMetrics::periods)
        .def_readonly("total_return", &PerformanceMetrics::total_return)
        .def_readonly("annualized_return", &PerformanceMetrics::annualized_return)
        .def_readonly("annualized_volatility", &PerformanceMetrics::annualized_volatility)
        .def_readonly("sharpe_ratio", &PerformanceMetrics::sharpe_ratio)
        .def_readonly("sortino_ratio", &PerformanceMetrics::sortino_ratio)
        .def_readonly("max_drawdown", &PerformanceMetrics::max_drawdown)
        .def_readonly("max_drawdown_duration", &PerformanceMetrics::max_drawdown_duration)
        .def_readonly("calmar_ratio", &PerformanceMetrics::calmar_ratio)
        .def_readonly("hit_rate", &PerformanceMetrics::hit_rate)
        .def_readonly("best_period", &PerformanceMetrics::best_period)
        .def_readonly("worst_period", &PerformanceMetrics::worst_period)
        .def_readonly("rebalances", &PerformanceMetrics::rebalances)
        .def_readonly("average_turnover", &PerformanceMetrics::average_turnover)
        .def_readonly("annualized_turnover", &PerformanceMetrics::annualized_turnover)
        .def_readonly("total_costs", &PerformanceMetrics::total_costs);
    m.def("compute_performance", &compute_performance, py::arg("returns"), py::arg("periods_per_year") = 252,
          "Metrics of a series of simple periodic returns.");
    m.def("equity_curve", &equity_curve, py::arg("returns"), py::arg("start") = 1.0, "start * cumprod(1 + r).");
    m.def("drawdown_series", &drawdown_series, py::arg("equity"), py::arg("start"),
          "1 - equity / running peak, with `start` as the first peak.");

    py::enum_<PortfolioRiskMeasure>(m, "PortfolioRiskMeasure")
        .value("Volatility", PortfolioRiskMeasure::Volatility)
        .value("ValueAtRisk", PortfolioRiskMeasure::ValueAtRisk)
        .value("ExpectedShortfall", PortfolioRiskMeasure::ExpectedShortfall);
    py::enum_<CovarianceEstimator>(m, "CovarianceEstimator")
        .value("Sample", CovarianceEstimator::Sample)
        .value("Exponential", CovarianceEstimator::Exponential);

    using P = RiskBudgetingPortfolioParams;
    py::class_<P>(m, "RiskBudgetingPortfolioParams")
        .def(kwargs_init<P>())
        .def_readwrite("covariance_window", &P::covariance_window)
        .def_readwrite("covariance_estimator", &P::covariance_estimator)
        .def_readwrite("ewma_halflife", &P::ewma_halflife)
        .def_readwrite("budgets", &P::budgets)
        .def_readwrite("risk_measure", &P::risk_measure)
        .def_readwrite("confidence_level", &P::confidence_level)
        .def_readwrite("use_expected_returns", &P::use_expected_returns)
        .def_readwrite("target_volatility", &P::target_volatility)
        .def_readwrite("max_leverage", &P::max_leverage)
        .def_readwrite("rebalance_frequency", &P::rebalance_frequency)
        .def_readwrite("transaction_cost", &P::transaction_cost)
        .def_readwrite("drift_weights", &P::drift_weights)
        .def_readwrite("initial_capital", &P::initial_capital)
        .def_readwrite("periods_per_year", &P::periods_per_year)
        .def_readwrite("solver_tolerance", &P::solver_tolerance)
        .def_readwrite("solver_max_iter", &P::solver_max_iter)
        .def_readwrite("num_threads", &P::num_threads);

    using C = PortfolioComputation;
    py::class_<C>(m, "PortfolioComputation", "Backtest output, one row per input row (R rows per rebalance).")
        .def_readonly("weights", &C::weights)
        .def_readonly("rebalance_rows", &C::rebalance_rows)
        .def_readonly("target_weights", &C::target_weights)
        .def_readonly("risk_contributions", &C::risk_contributions)
        .def_readonly("leverage", &C::leverage)
        .def_readonly("gross_returns", &C::gross_returns)
        .def_readonly("costs", &C::costs)
        .def_readonly("net_returns", &C::net_returns)
        .def_readonly("turnover", &C::turnover)
        .def_readonly("equity", &C::equity)
        .def_readonly("pnl", &C::pnl)
        .def_readonly("drawdown", &C::drawdown)
        .def_readonly("metrics", &C::metrics)
        .def_readonly("all_converged", &C::all_converged)
        .def_readonly("threads_used", &C::threads_used);

    py::class_<RiskBudgetingPortfolio>(m, "RiskBudgetingPortfolio", "Walk-forward risk budgeting backtest.")
        .def(py::init([](const P& p) { return std::make_unique<RiskBudgetingPortfolio>(std::make_unique<P>(p)); }),
             py::arg("parameters"))
        .def(py::init([](const py::kwargs& kw) { return std::make_unique<RiskBudgetingPortfolio>(from_kwargs<P>(kw)); }))
        .def(
            "run",
            [](const RiskBudgetingPortfolio& engine, const MatrixXd& returns,
               const std::optional<std::vector<Index>>& rows, const std::optional<MatrixXd>& budgets) {
                const std::vector<sd::BudgetChange> changes = schedule(rows, budgets, returns.cols());
                const py::gil_scoped_release release;
                return sd::run_risk_budgeting_portfolio(engine.parameters(), returns, changes);
            },
            py::arg("returns"), py::arg("schedule_rows") = py::none(), py::arg("schedule_budgets") = py::none(),
            "returns: (T x N) simple returns, rows in time order. Optional schedule: budgets row k applies to the\n"
            "rebalances from returns row schedule_rows[k] on (rows sorted).")
        .def_property_readonly("parameters", &RiskBudgetingPortfolio::parameters);
}

} // namespace nabia_bridge
