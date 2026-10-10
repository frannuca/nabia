//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/strategy/PerformanceMetrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Eigen::VectorXd equity_curve(const Eigen::VectorXd& returns, double start) {
    Eigen::VectorXd equity(returns.size());
    double e = start;
    for (Eigen::Index t = 0; t < returns.size(); ++t) {
        e *= 1.0 + returns(t);
        equity(t) = e;
    }
    return equity;
}

Eigen::VectorXd drawdown_series(const Eigen::VectorXd& equity, double start) {
    Eigen::VectorXd dd(equity.size());
    double peak = start;
    for (Eigen::Index t = 0; t < equity.size(); ++t) {
        peak = std::max(peak, equity(t));
        dd(t) = peak > 0.0 ? 1.0 - equity(t) / peak : 0.0;
    }
    return dd;
}

PerformanceMetrics compute_performance(const Eigen::VectorXd& returns, int periods_per_year) {
    if (returns.size() == 0 || !returns.allFinite()) {
        throw std::invalid_argument("compute_performance: returns must be non-empty and finite");
    }
    if (periods_per_year <= 0) {
        throw std::invalid_argument("compute_performance: periods_per_year must be positive");
    }
    const auto n = static_cast<double>(returns.size());
    const double ppy = static_cast<double>(periods_per_year);
    PerformanceMetrics m;
    m.periods = returns.size();

    const Eigen::VectorXd equity = equity_curve(returns, 1.0);
    m.total_return = equity(equity.size() - 1) - 1.0;
    m.annualized_return = m.total_return > -1.0 ? std::pow(1.0 + m.total_return, ppy / n) - 1.0 : -1.0;

    const double mean = returns.mean();
    const double var = returns.size() > 1 ? (returns.array() - mean).square().sum() / (n - 1.0) : 0.0;
    m.annualized_volatility = std::sqrt(var * ppy);
    m.sharpe_ratio = m.annualized_volatility > 0.0 ? mean * ppy / m.annualized_volatility : 0.0;
    const double downside = std::sqrt(returns.cwiseMin(0.0).squaredNorm() / n) * std::sqrt(ppy);
    m.sortino_ratio = downside > 0.0 ? mean * ppy / downside : 0.0;

    const Eigen::VectorXd dd = drawdown_series(equity, 1.0);
    m.max_drawdown = dd.maxCoeff();
    Eigen::Index run = 0;
    for (Eigen::Index t = 0; t < dd.size(); ++t) {
        run = dd(t) > 0.0 ? run + 1 : 0;
        m.max_drawdown_duration = std::max(m.max_drawdown_duration, run);
    }
    m.calmar_ratio = m.max_drawdown > 0.0 ? m.annualized_return / m.max_drawdown : 0.0;

    m.hit_rate = static_cast<double>((returns.array() > 0.0).count()) / n;
    m.best_period = returns.maxCoeff();
    m.worst_period = returns.minCoeff();
    return m;
}
