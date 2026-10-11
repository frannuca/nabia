//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_RISKBUDGETINGPORTFOLIO_H
#define NABIA_RISKBUDGETINGPORTFOLIO_H
#include "nabia/dataframe/DataFrame.h"
#include "nabia/dataframe/TimeSeries.h"
#include "nabia/strategy/PerformanceMetrics.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Eigen/Dense>

/// Risk measure whose contributions are budgeted (see nabia/risk).
enum class PortfolioRiskMeasure {
    Volatility,
    ValueAtRisk,        // Gaussian, at confidence_level
    ExpectedShortfall,  // Gaussian, at confidence_level
};

/// How the covariance is estimated from the trailing window.
enum class CovarianceEstimator {
    Sample,       // equally weighted sample covariance
    Exponential,  // exponentially weighted, half-life ewma_halflife points (recent data count more)
};

struct RiskBudgetingPortfolioParams {
    // --- Estimation ---
    Eigen::Index covariance_window = 60;     // points of the input series used for each covariance (>= 2)
    CovarianceEstimator covariance_estimator = CovarianceEstimator::Sample;
    double ewma_halflife = 20.0;             // in points, for CovarianceEstimator::Exponential
    // --- Allocation ---
    Eigen::VectorXd budgets;                 // risk budgets per asset (> 0, normalized); empty = equal risk contribution
    PortfolioRiskMeasure risk_measure = PortfolioRiskMeasure::Volatility;
    double confidence_level = 0.99;          // for ValueAtRisk / ExpectedShortfall, in (0.5, 1)
    bool use_expected_returns = false;       // VaR/ES: use the window mean (noisy) instead of zero
    std::optional<double> target_volatility; // annualized; scales the gross exposure to this predicted volatility
    double max_leverage = 2.0;               // cap on sum(weights) when targeting volatility
    // --- Execution ---
    Eigen::Index rebalance_frequency = 1;    // trade every k points: 1 = every point, 2 = every other point, ...
    double transaction_cost = 0.0;           // proportional cost per unit of traded notional (0.0005 = 5 bp)
    bool drift_weights = true;               // between trades, weights drift with asset returns (else held constant)
    // --- Accounting ---
    double initial_capital = 1.0;
    int periods_per_year = 252;              // for annualized metrics and volatility targeting
    // --- Solver ---
    double solver_tolerance = 1e-8;          // risk budgeting accuracy, max |RC_i / R - b_i|
    int solver_max_iter = 5000;
    /// Threads solving the risk budgeting problems of the rebalance dates in
    /// parallel, the calling thread included. 0 = std::thread::hardware_concurrency(),
    /// 1 = serial. Results are identical for every thread count.
    int num_threads = 0;
};

/// Index-free output of the engine, one row (or entry) per input point.
struct PortfolioComputation {
    Eigen::MatrixXd weights;                 // T x N, held over each period
    std::vector<Eigen::Index> rebalance_rows;// rows at whose close a trade happened
    Eigen::MatrixXd target_weights;          // R x N, weights set at each rebalance
    Eigen::MatrixXd risk_contributions;      // R x N, relative risk contributions at each rebalance
    Eigen::VectorXd leverage;                // R, sum of target weights
    Eigen::VectorXd gross_returns;           // T, w . r before costs
    Eigen::VectorXd costs;                   // T, cost as a fraction of equity
    Eigen::VectorXd net_returns;             // T, (1 + gross)(1 - cost) - 1
    Eigen::VectorXd turnover;                // T, sum |w_new - w_old| traded at the close
    Eigen::VectorXd equity;                  // T, capital after each period
    Eigen::VectorXd pnl;                     // T, change in capital
    Eigen::VectorXd drawdown;                // T, 1 - equity / running peak
    PerformanceMetrics metrics;              // over the active period (from the first trade)
    bool all_converged = true;               // every risk budgeting solve met solver_tolerance
    int threads_used = 1;                    // threads that solved the risk budgeting problems
};

/// Result of a backtest, with the time index of the input returns.
template <TimeIndex Index>
struct PortfolioResult {
    DataFrame<Index> weights;            // weights held over each period (row t earns the returns of row t)
    DataFrame<Index> target_weights;     // at each rebalance time, after volatility targeting
    DataFrame<Index> risk_contributions; // relative risk contributions at each rebalance (sum to 1)
    TimeSeries<Index> leverage;          // sum of target weights at each rebalance
    TimeSeries<Index> gross_returns;
    TimeSeries<Index> costs;
    TimeSeries<Index> net_returns;       // the strategy's periodic return (PnL in return units)
    TimeSeries<Index> turnover;
    TimeSeries<Index> equity;
    TimeSeries<Index> pnl;               // PnL in currency units
    TimeSeries<Index> drawdown;
    PerformanceMetrics metrics;
    bool all_converged = true;
    int threads_used = 1;
};

namespace nabia::strategy::detail {
/// Budgets that take effect from a given row of the returns on.
struct BudgetChange {
    Eigen::Index first_row;   // first returns row whose rebalance uses these budgets
    Eigen::VectorXd budgets;
};

/// Non-template engine; `returns` is T x N (rows = time). `schedule` (sorted
/// by first_row, may be empty) overrides params.budgets from each first_row on.
PortfolioComputation run_risk_budgeting_portfolio(const RiskBudgetingPortfolioParams& params,
                                                  const Eigen::MatrixXd& returns,
                                                  const std::vector<BudgetChange>& schedule = {});
void validate(const RiskBudgetingPortfolioParams& params);
} // namespace nabia::strategy::detail

/// Walk-forward backtest of a long-only risk budgeting portfolio.
///
/// Input: simple returns per period, one column per asset, rows ordered in time
/// (row t holds the returns over (t-1, t]). At the close of every rebalance
/// point t (t = W-1, W-1+k, W-1+2k, ... for window W and frequency k) the engine
///   1. estimates the covariance from rows t-W+1..t (only data known at t),
///   2. solves for the weights whose contributions to the chosen risk measure
///      match the budgets (RiskBudgeting), optionally scaled to a target volatility,
///   3. trades from the current (drifted) weights to the targets, paying
///      transaction_cost * turnover as a fraction of equity.
/// The new weights earn the returns from row t+1 on, so there is no look-ahead.
/// Between rebalances the weights drift with returns (or stay constant if
/// drift_weights is false); any weight not invested (1 - sum w) is cash earning 0.
///
/// The target weights of a rebalance depend only on its own window, so steps 1-2
/// run for all rebalance dates first, in parallel on num_threads threads; step 3
/// (drift, trading and accounting) then runs sequentially through time. Each
/// solve warm-starts from the previous rebalance within fixed blocks of
/// consecutive dates (one block per task), so the results do not depend on the
/// number of threads.
class RiskBudgetingPortfolio {
public:
    /// Throws std::invalid_argument on invalid parameters.
    explicit RiskBudgetingPortfolio(std::unique_ptr<RiskBudgetingPortfolioParams> parameters)
        : parameters_(std::move(parameters)) {
        if (!parameters_) {
            throw std::invalid_argument("RiskBudgetingPortfolio: parameters are null");
        }
        nabia::strategy::detail::validate(*parameters_);
    }

    /// Runs the backtest. Throws std::invalid_argument if the returns have fewer
    /// rows than the covariance window, non-finite values, or a column count
    /// different from the budgets.
    template <TimeIndex Index>
    [[nodiscard]] PortfolioResult<Index> run(const DataFrame<Index>& returns) const {
        return make_result(returns, nabia::strategy::detail::run_risk_budgeting_portfolio(*parameters_, returns.values()));
    }

    /// Backtest with risk budgets that change over time. Row d of
    /// `budget_schedule` (one column per asset, same columns as `returns`)
    /// holds the budgets decided at time d: they are used by every rebalance at
    /// or after d, until the next row. Rebalances before the first row use
    /// parameters().budgets (equal risk if empty). Budgets are normalized per
    /// row. Throws std::invalid_argument if the columns differ or a budget is
    /// not positive.
    template <TimeIndex Index>
    [[nodiscard]] PortfolioResult<Index> run(const DataFrame<Index>& returns, const DataFrame<Index>& budget_schedule) const {
        if (budget_schedule.columns() != returns.columns()) {
            throw std::invalid_argument("RiskBudgetingPortfolio: budget schedule columns must match the returns columns");
        }
        std::vector<nabia::strategy::detail::BudgetChange> schedule;
        schedule.reserve(static_cast<std::size_t>(budget_schedule.rows()));
        const std::vector<Index>& index = returns.index();
        for (Eigen::Index s = 0; s < budget_schedule.rows(); ++s) {
            const auto first = std::lower_bound(index.begin(), index.end(), budget_schedule.time(s));
            schedule.push_back({static_cast<Eigen::Index>(first - index.begin()), budget_schedule.values().row(s).transpose()});
        }
        return make_result(returns,
                           nabia::strategy::detail::run_risk_budgeting_portfolio(*parameters_, returns.values(), schedule));
    }

    [[nodiscard]] const RiskBudgetingPortfolioParams& parameters() const { return *parameters_; }

private:
    template <TimeIndex Index>
    static PortfolioResult<Index> make_result(const DataFrame<Index>& returns, PortfolioComputation c) {
        const std::vector<Index>& index = returns.index();
        std::vector<Index> rebalance_index;
        rebalance_index.reserve(c.rebalance_rows.size());
        for (Eigen::Index row : c.rebalance_rows) {
            rebalance_index.push_back(index[static_cast<std::size_t>(row)]);
        }
        PortfolioResult<Index> r;
        r.weights = DataFrame<Index>(index, returns.columns(), std::move(c.weights));
        r.target_weights = DataFrame<Index>(rebalance_index, returns.columns(), std::move(c.target_weights));
        r.risk_contributions = DataFrame<Index>(rebalance_index, returns.columns(), std::move(c.risk_contributions));
        r.leverage = TimeSeries<Index>(rebalance_index, std::move(c.leverage), "leverage");
        r.gross_returns = TimeSeries<Index>(index, std::move(c.gross_returns), "gross_returns");
        r.costs = TimeSeries<Index>(index, std::move(c.costs), "costs");
        r.net_returns = TimeSeries<Index>(index, std::move(c.net_returns), "net_returns");
        r.turnover = TimeSeries<Index>(index, std::move(c.turnover), "turnover");
        r.equity = TimeSeries<Index>(index, std::move(c.equity), "equity");
        r.pnl = TimeSeries<Index>(index, std::move(c.pnl), "pnl");
        r.drawdown = TimeSeries<Index>(index, std::move(c.drawdown), "drawdown");
        r.metrics = c.metrics;
        r.all_converged = c.all_converged;
        r.threads_used = c.threads_used;
        return r;
    }

    std::unique_ptr<RiskBudgetingPortfolioParams> parameters_;
};

#endif // NABIA_RISKBUDGETINGPORTFOLIO_H
