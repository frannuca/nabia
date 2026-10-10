//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_PERFORMANCEMETRICS_H
#define NABIA_PERFORMANCEMETRICS_H
#include <Eigen/Dense>

/// Summary statistics of a series of periodic (e.g. daily) simple returns.
/// Annualized figures assume `periods_per_year` periods a year and a zero
/// risk-free rate.
struct PerformanceMetrics {
    Eigen::Index periods = 0;             // number of returns used
    double total_return = 0.0;            // prod(1 + r) - 1
    double annualized_return = 0.0;       // (1 + total)^(periods_per_year / periods) - 1 (CAGR)
    double annualized_volatility = 0.0;   // std(r, ddof = 1) * sqrt(periods_per_year)
    double sharpe_ratio = 0.0;            // mean(r) * periods_per_year / annualized_volatility
    double sortino_ratio = 0.0;           // mean(r) * periods_per_year / (sqrt(mean(min(r, 0)^2)) * sqrt(periods_per_year))
    double max_drawdown = 0.0;            // largest peak-to-trough loss of the equity curve, as a positive fraction
    Eigen::Index max_drawdown_duration = 0;  // longest stretch of periods spent below a previous peak
    double calmar_ratio = 0.0;            // annualized_return / max_drawdown
    double hit_rate = 0.0;                // fraction of periods with r > 0
    double best_period = 0.0;             // max r
    double worst_period = 0.0;            // min r
    // Trading statistics, filled by the portfolio engines (zero otherwise).
    Eigen::Index rebalances = 0;
    double average_turnover = 0.0;        // mean sum|w_new - w_old| per rebalance
    double annualized_turnover = 0.0;     // total turnover * periods_per_year / periods
    double total_costs = 0.0;             // sum of transaction costs, as a fraction of equity
};

/// Metrics of `returns` (simple returns per period). Ratios whose denominator
/// is zero are reported as 0. Throws std::invalid_argument if returns is empty,
/// not finite, or periods_per_year <= 0.
[[nodiscard]] PerformanceMetrics compute_performance(const Eigen::VectorXd& returns, int periods_per_year);

/// Equity curve start * prod(1 + r) after each period.
[[nodiscard]] Eigen::VectorXd equity_curve(const Eigen::VectorXd& returns, double start = 1.0);

/// Drawdown after each period: 1 - equity / running peak (0 at a new high).
/// The starting value `start` counts as the first peak.
[[nodiscard]] Eigen::VectorXd drawdown_series(const Eigen::VectorXd& equity, double start);

#endif // NABIA_PERFORMANCEMETRICS_H
