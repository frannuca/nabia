//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/risk/RiskBudgeting.h"

#include "nabia/optimization/convex/LBFGS.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nabia::risk::detail {

using Eigen::VectorXd;

namespace {

VectorXd normalized_budgets(const RiskBudgetingParams& params, Eigen::Index n) {
    if (params.budgets.size() == 0) {
        return VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    }
    if (params.budgets.size() != n) {
        throw std::invalid_argument("RiskBudgeting: budgets must have one entry per asset");
    }
    if (!params.budgets.allFinite() || (params.budgets.array() <= 0.0).any()) {
        throw std::invalid_argument("RiskBudgeting: budgets must be positive");
    }
    return params.budgets / params.budgets.sum();
}

/// Default start: y_i proportional to b_i / sigma_i (exact for uncorrelated
/// assets under volatility).
VectorXd default_start(const IRiskMeasure& measure, const VectorXd& b) {
    const VectorXd sigma = measure.covariance_matrix().diagonal().cwiseSqrt();
    VectorXd y(b.size());
    for (Eigen::Index i = 0; i < b.size(); ++i) {
        y(i) = sigma(i) > 0.0 ? b(i) / sigma(i) : b(i);
    }
    return y;
}

} // namespace

RiskBudgetingResult solve_risk_budgeting(const IRiskMeasure& measure, const RiskBudgetingParams& params,
                                         std::optional<VectorXd> initial_guess) {
    const Eigen::Index n = measure.dim();
    const VectorXd b = normalized_budgets(params, n);
    if (!(params.tolerance > 0.0)) {
        throw std::invalid_argument("RiskBudgeting: tolerance must be positive");
    }

    VectorXd y0 = initial_guess ? std::move(*initial_guess) : default_start(measure, b);
    if (y0.size() != n || !y0.allFinite() || (y0.array() <= 0.0).any()) {
        throw std::invalid_argument("RiskBudgeting: initial guess must have one positive entry per asset");
    }
    // Scale so that R(y0) = 1 = sum(b): the solution of the log-barrier problem
    // has R(y*) = sum(b) (Euler), so this starts at the right magnitude.
    const double r0 = measure.compute_risk(y0);
    if (!(r0 > 0.0)) {
        throw std::invalid_argument("RiskBudgeting: risk must be positive on long-only portfolios");
    }
    y0 /= r0;

    // f(y) = R(y) - b' log(y),  grad f(y) = dR/dy - b ./ y,  y_i grad_i = RC_i(y) - b_i.
    // Points with some y_i <= 0 get +inf, which the line search rejects.
    const LBFGS::ValueAndGradient objective = [&](const VectorXd& y, VectorXd& grad) {
        if ((y.array() <= 0.0).any()) {
            grad.setZero();
            return std::numeric_limits<double>::infinity();
        }
        grad = measure.compute_marginal_risk_contributions(y) - b.cwiseQuotient(y);
        return measure.compute_risk(y) - b.dot(VectorXd(y.array().log()));
    };
    auto budget_error = [&](const VectorXd& y) {
        return (measure.compute_risk_contributions(y) / measure.compute_risk(y) - b).lpNorm<Eigen::Infinity>();
    };

    // L-BFGS stops on ||grad||_inf, but the budget error is ~ ||y .* grad||_inf,
    // so start from tolerance / max(y) and, if needed, warm-restart with a
    // tighter tolerance until the budget error itself is below `tolerance`.
    VectorXd y = y0;
    double inner_tol = params.tolerance / y0.maxCoeff();
    int iterations = 0;
    constexpr int max_restarts = 5;
    for (int restart = 0; restart <= max_restarts && iterations < params.max_iter; ++restart) {
        LBFGSParams lbfgs_params{n};
        lbfgs_params.tolerance = inner_tol;
        lbfgs_params.max_iter = params.max_iter - iterations;
        lbfgs_params.memory = params.memory;
        LBFGS solver(std::make_unique<LBFGSParams>(lbfgs_params), objective);
        y = solver.optimize(y);
        iterations += solver.result().iterations;

        const double err = budget_error(y);
        if (err <= params.tolerance || solver.result().status == LBFGSStatus::LineSearchFailed) {
            break;
        }
        inner_tol *= 0.5 * params.tolerance / err;
    }

    RiskBudgetingResult result;
    result.weights = y / y.sum();
    result.risk = measure.compute_risk(result.weights);
    result.risk_contributions = measure.compute_risk_contributions(result.weights);
    result.relative_contributions = result.risk_contributions / result.risk;
    result.budget_error = (result.relative_contributions - b).lpNorm<Eigen::Infinity>();
    result.iterations = iterations;
    result.converged = std::isfinite(result.budget_error) && result.budget_error <= params.tolerance;
    return result;
}

} // namespace nabia::risk::detail
