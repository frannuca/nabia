//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/strategy/RiskBudgetingPortfolio.h"

#include "nabia/risk/RiskBudgeting.h"
#include "nabia/risk/measures/ExpectedShortfall.h"
#include "nabia/risk/measures/ValueAtRisk.h"
#include "nabia/risk/measures/Volatility.h"

#include "../common/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace nabia::strategy::detail {

namespace {

struct WindowMoments {
    VectorXd mean;
    MatrixXd covariance;
};

/// Mean and covariance of the rows of `window` (oldest first).
WindowMoments estimate(const RiskBudgetingPortfolioParams& p, const MatrixXd& window) {
    const Index W = window.rows();
    VectorXd w = VectorXd::Constant(W, 1.0 / static_cast<double>(W));
    if (p.covariance_estimator == CovarianceEstimator::Exponential) {
        // Weight lambda^age with lambda = 2^(-1 / halflife); the newest row has age 0.
        const double lambda = std::exp2(-1.0 / p.ewma_halflife);
        for (Index i = 0; i < W; ++i) {
            w(i) = std::pow(lambda, static_cast<double>(W - 1 - i));
        }
        w /= w.sum();
    }
    WindowMoments m;
    m.mean = window.transpose() * w;
    const MatrixXd centred = window.rowwise() - m.mean.transpose();
    // Unbiased weighted covariance: divide by 1 - sum(w^2) (= (W - 1) / W for equal weights).
    m.covariance = centred.transpose() * w.asDiagonal() * centred / (1.0 - w.squaredNorm());
    m.covariance = 0.5 * (m.covariance + m.covariance.transpose());
    return m;
}

template <typename Measure>
RiskBudgetingResult solve(const RiskBudgetingPortfolioParams& p, const VectorXd& budgets, Measure measure,
                          const std::optional<VectorXd>& warm_start) {
    auto params = std::make_unique<RiskBudgetingParams>();
    params->budgets = budgets;
    params->tolerance = p.solver_tolerance;
    params->max_iter = p.solver_max_iter;
    RiskBudgeting<Measure> rb(std::move(params), std::move(measure));
    rb.optimize(warm_start);
    return rb.result();
}

RiskBudgetingResult allocate(const RiskBudgetingPortfolioParams& p, const VectorXd& budgets, const WindowMoments& m,
                             const std::optional<VectorXd>& warm_start) {
    const VectorXd mu = p.use_expected_returns ? m.mean : VectorXd::Zero(m.mean.size());
    switch (p.risk_measure) {
    case PortfolioRiskMeasure::Volatility:
        return solve(p, budgets, Volatility(m.covariance), warm_start);
    case PortfolioRiskMeasure::ValueAtRisk:
        return solve(p, budgets, ValueAtRisk(m.covariance, p.confidence_level, mu), warm_start);
    case PortfolioRiskMeasure::ExpectedShortfall:
        return solve(p, budgets, ExpectedShortfall(m.covariance, p.confidence_level, mu), warm_start);
    }
    throw std::invalid_argument("RiskBudgetingPortfolio: unknown risk measure");
}

/// Consecutive rebalance dates solved by one task, each warm-started from the
/// previous one. Fixed, so results do not depend on the number of threads.
constexpr Eigen::Index kWarmStartBlock = 8;

struct Target {
    VectorXd weights;        // after volatility targeting
    VectorXd contributions;  // relative risk contributions
    double leverage = 1.0;
    bool converged = true;
};

Target solve_target(const RiskBudgetingPortfolioParams& p, const VectorXd& budgets, const MatrixXd& window,
                    const std::optional<VectorXd>& warm_start) {
    const WindowMoments m = estimate(p, window);
    const RiskBudgetingResult rb = allocate(p, budgets, m, warm_start);
    Target target;
    if (p.target_volatility) {
        const double ppy = static_cast<double>(p.periods_per_year);
        const double predicted = std::sqrt(std::max(0.0, rb.weights.dot(m.covariance * rb.weights)) * ppy);
        target.leverage = predicted > 0.0 ? std::min(p.max_leverage, *p.target_volatility / predicted) : p.max_leverage;
    }
    target.weights = target.leverage * rb.weights;
    target.contributions = rb.relative_contributions;
    target.converged = rb.converged;
    return target;
}

} // namespace

void validate(const RiskBudgetingPortfolioParams& p) {
    if (p.covariance_window < 2) {
        throw std::invalid_argument("RiskBudgetingPortfolio: covariance_window must be >= 2");
    }
    if (p.rebalance_frequency < 1) {
        throw std::invalid_argument("RiskBudgetingPortfolio: rebalance_frequency must be >= 1");
    }
    if (!(p.transaction_cost >= 0.0 && p.transaction_cost < 1.0)) {
        throw std::invalid_argument("RiskBudgetingPortfolio: transaction_cost must be in [0, 1)");
    }
    if (p.covariance_estimator == CovarianceEstimator::Exponential && !(p.ewma_halflife > 0.0)) {
        throw std::invalid_argument("RiskBudgetingPortfolio: ewma_halflife must be positive");
    }
    if (p.risk_measure != PortfolioRiskMeasure::Volatility &&
        !(p.confidence_level > 0.5 && p.confidence_level < 1.0)) {
        throw std::invalid_argument("RiskBudgetingPortfolio: confidence_level must be in (0.5, 1)");
    }
    if (p.target_volatility && !(*p.target_volatility > 0.0)) {
        throw std::invalid_argument("RiskBudgetingPortfolio: target_volatility must be positive");
    }
    if (p.num_threads < 0) {
        throw std::invalid_argument("RiskBudgetingPortfolio: num_threads must be >= 0");
    }
    if (!(p.max_leverage > 0.0) || !(p.initial_capital > 0.0) || p.periods_per_year <= 0) {
        throw std::invalid_argument("RiskBudgetingPortfolio: max_leverage, initial_capital and periods_per_year must be positive");
    }
}

PortfolioComputation run_risk_budgeting_portfolio(const RiskBudgetingPortfolioParams& p, const MatrixXd& returns,
                                                  const std::vector<BudgetChange>& schedule) {
    validate(p);
    const Index T = returns.rows();
    const Index N = returns.cols();
    const Index W = p.covariance_window;
    if (N == 0 || T < W) {
        throw std::invalid_argument("RiskBudgetingPortfolio: need at least one asset and covariance_window rows");
    }
    if (!returns.allFinite()) {
        throw std::invalid_argument("RiskBudgetingPortfolio: returns must be finite");
    }
    if (p.budgets.size() != 0 && p.budgets.size() != N) {
        throw std::invalid_argument("RiskBudgetingPortfolio: budgets must have one entry per asset");
    }
    const VectorXd default_budgets =
        p.budgets.size() == 0 ? VectorXd::Constant(N, 1.0 / static_cast<double>(N)) : p.budgets;
    for (std::size_t s = 0; s < schedule.size(); ++s) {
        const VectorXd& b = schedule[s].budgets;
        if (b.size() != N || !b.allFinite() || (b.array() <= 0.0).any()) {
            throw std::invalid_argument("RiskBudgetingPortfolio: scheduled budgets must be positive, one per asset");
        }
        if (s > 0 && schedule[s].first_row < schedule[s - 1].first_row) {
            throw std::invalid_argument("RiskBudgetingPortfolio: budget schedule must be in time order");
        }
    }
    // Budgets in force at a rebalance row: the last change at or before it.
    auto budgets_at = [&](Index row) -> const VectorXd& {
        const VectorXd* b = &default_budgets;
        for (const BudgetChange& change : schedule) {
            if (change.first_row > row) {
                break;
            }
            b = &change.budgets;
        }
        return *b;
    };

    PortfolioComputation c;
    c.weights = MatrixXd::Zero(T, N);
    c.gross_returns = VectorXd::Zero(T);
    c.costs = VectorXd::Zero(T);
    c.net_returns = VectorXd::Zero(T);
    c.turnover = VectorXd::Zero(T);
    c.equity = VectorXd::Zero(T);
    c.pnl = VectorXd::Zero(T);
    const Index n_rebalances = (T - W) / p.rebalance_frequency + 1;
    c.target_weights.resize(n_rebalances, N);
    c.risk_contributions.resize(n_rebalances, N);
    c.leverage.resize(n_rebalances);
    for (Index k = 0; k < n_rebalances; ++k) {
        c.rebalance_rows.push_back(W - 1 + k * p.rebalance_frequency);
    }

    // Phase 1 (parallel): target weights of every rebalance date. Each task
    // solves one block of consecutive dates, warm-starting within the block.
    std::vector<Target> targets(static_cast<std::size_t>(n_rebalances));
    const Index n_blocks = (n_rebalances + kWarmStartBlock - 1) / kWarmStartBlock;
    const int threads = nabia::detail::resolve_threads(p.num_threads, n_blocks);
    nabia::detail::parallel_for(n_blocks, threads, [&](std::int64_t block) {
        std::optional<VectorXd> warm_start;
        const Index first = block * kWarmStartBlock;
        const Index last = std::min(n_rebalances, first + kWarmStartBlock);
        for (Index k = first; k < last; ++k) {
            const Index t = c.rebalance_rows[static_cast<std::size_t>(k)];
            Target target = solve_target(p, budgets_at(t), returns.middleRows(t - W + 1, W), warm_start);
            warm_start = target.weights / target.leverage;  // the unlevered risk budgeting weights
            targets[static_cast<std::size_t>(k)] = std::move(target);
        }
    });
    c.threads_used = threads;

    // Phase 2 (sequential): drift, trade and account through time.
    VectorXd w = VectorXd::Zero(N);  // current weights (fractions of equity)
    double equity = p.initial_capital;
    const double ppy = static_cast<double>(p.periods_per_year);
    Index next = 0;  // next rebalance

    for (Index t = 0; t < T; ++t) {
        // 1. Earn this period's returns with the weights set at the previous close.
        c.weights.row(t) = w.transpose();
        const VectorXd r = returns.row(t).transpose();
        const double gross = w.dot(r);
        if (!(1.0 + gross > 0.0)) {
            throw std::invalid_argument("RiskBudgetingPortfolio: the portfolio lost all its capital at row " +
                                        std::to_string(t));
        }
        if (p.drift_weights) {
            // Each holding grows by (1 + r_i), equity by (1 + gross); cash is the remainder.
            w = w.cwiseProduct((VectorXd::Ones(N) + r)) / (1.0 + gross);
        }

        // 2. Rebalance at the close, with the data known at t.
        double cost = 0.0;
        if (next < n_rebalances && c.rebalance_rows[static_cast<std::size_t>(next)] == t) {
            const Target& target = targets[static_cast<std::size_t>(next)];
            const double traded = (target.weights - w).lpNorm<1>();
            cost = p.transaction_cost * traded;
            w = target.weights;

            c.target_weights.row(next) = target.weights.transpose();
            c.risk_contributions.row(next) = target.contributions.transpose();
            c.leverage(next) = target.leverage;
            c.all_converged = c.all_converged && target.converged;
            c.turnover(t) = traded;
            ++next;
        }

        // 3. Accounting: costs are paid out of the equity at the close.
        const double net = (1.0 + gross) * (1.0 - cost) - 1.0;
        const double new_equity = equity * (1.0 + net);
        c.gross_returns(t) = gross;
        c.costs(t) = cost;
        c.net_returns(t) = net;
        c.pnl(t) = new_equity - equity;
        c.equity(t) = new_equity;
        equity = new_equity;
    }

    c.drawdown = drawdown_series(c.equity, p.initial_capital);

    // Metrics from the first trade on (the rows before it hold no positions).
    const Index first = W - 1;
    c.metrics = compute_performance(c.net_returns.tail(T - first), p.periods_per_year);
    c.metrics.rebalances = n_rebalances;
    c.metrics.average_turnover = c.turnover.sum() / static_cast<double>(n_rebalances);
    c.metrics.annualized_turnover = c.turnover.sum() * ppy / static_cast<double>(T - first);
    c.metrics.total_costs = c.costs.sum();
    return c;
}

} // namespace nabia::strategy::detail
