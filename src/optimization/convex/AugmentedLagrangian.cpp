//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/convex/AugmentedLagrangian.h"
#include "nabia/optimization/convex/LBFGS.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace

AugmentedLagrangian::AugmentedLagrangian(std::unique_ptr<ConvexOptimizationParams> parameters,
                                     Objective objective,
                                     Gradient gradient,
                                     AugmentedLagrangianOptions options)
    : IConvexOptimization(std::move(parameters)),
      objective_(std::move(objective)),
      gradient_(std::move(gradient)),
      options_(options) {}

Eigen::VectorXd AugmentedLagrangian::optimize(std::optional<Eigen::VectorXd> initial_guess) {
    if (!parameters) {
        throw std::invalid_argument("AugmentedLagrangian: parameters are null");
    }
    if (!objective_ || !gradient_) {
        throw std::invalid_argument("AugmentedLagrangian: objective and gradient are required");
    }
    const ConvexOptimizationParams& p = *parameters;
    const Index n = p.dim;
    if (n <= 0 || (initial_guess && initial_guess->size() != n)) {
        throw std::invalid_argument("AugmentedLagrangian: `dim` must be > 0 and the initial guess must have `dim` entries");
    }
    if (!(p.tolerance > 0.0) || !(options_.initial_penalty > 0.0) || !(options_.penalty_growth > 1.0)) {
        throw std::invalid_argument("AugmentedLagrangian: tolerance, penalty and growth must be positive");
    }

    const LinearConstraint lf = to_linear_constraint(p.constraints, n);
    const bool has_eq = lf.b_eq.size() > 0;
    const bool has_ineq = lf.b_ineq.size() > 0;

    VectorXd x = initial_guess ? std::move(*initial_guess) : VectorXd::Zero(n);
    VectorXd lambda = VectorXd::Zero(lf.b_eq.size());
    VectorXd mu = VectorXd::Zero(lf.b_ineq.size());
    double rho = options_.initial_penalty;

    // L_rho(x) = f + lambda'h + rho/2 |h|^2 + 1/(2 rho) sum(max(0, mu + rho g)^2 - mu^2)
    // grad     = grad f + A_eq'(lambda + rho h) + A_ineq' max(0, mu + rho g)
    const LBFGS::ValueAndGradient augmented = [&](const VectorXd& z, VectorXd& grad) {
        double value = objective_(z);
        grad = gradient_(z);
        if (has_eq) {
            const VectorXd h = lf.A_eq * z - lf.b_eq;
            value += lambda.dot(h) + 0.5 * rho * h.squaredNorm();
            grad.noalias() += lf.A_eq.transpose() * (lambda + rho * h);
        }
        if (has_ineq) {
            const VectorXd shifted = (mu + rho * (lf.A_ineq * z - lf.b_ineq)).cwiseMax(0.0);
            value += (shifted.squaredNorm() - mu.squaredNorm()) / (2.0 * rho);
            grad.noalias() += lf.A_ineq.transpose() * shifted;
        }
        return value;
    };

    result_ = AugmentedLagrangianResult{};
    // Without constraints there is a single subproblem, so solve it to full accuracy.
    double inner_tol = (has_eq || has_ineq) ? std::max(p.tolerance, options_.initial_inner_tolerance) : p.tolerance;
    double previous_violation = kInf;
    const int max_outer = static_cast<int>(p.max_iter);

    for (int outer = 0; outer < max_outer; ++outer) {
        LBFGSParams inner_params{n};
        inner_params.tolerance = inner_tol;
        inner_params.max_iter = options_.max_inner_iter;
        inner_params.memory = options_.lbfgs_memory;
        LBFGS inner_solver(std::make_unique<LBFGSParams>(inner_params), augmented);
        x = inner_solver.optimize(x);  // warm start at the previous iterate
        const LBFGSResult& inner = inner_solver.result();
        result_.inner_iterations += inner.iterations;
        result_.outer_iterations = outer + 1;

        // Violation measured before the multiplier update; for inequalities
        // max(g, -mu/rho) is zero iff g <= 0 and mu * g = 0 (complementarity).
        double violation = 0.0;
        if (has_eq) {
            const VectorXd h = lf.A_eq * x - lf.b_eq;
            violation = h.lpNorm<Eigen::Infinity>();
            lambda += rho * h;
        }
        if (has_ineq) {
            const VectorXd g = lf.A_ineq * x - lf.b_ineq;
            violation = std::max(violation, g.cwiseMax(-mu / rho).lpNorm<Eigen::Infinity>());
            mu = (mu + rho * g).cwiseMax(0.0);
        }

        // After the update, grad L_rho(x) = grad f + A_eq' lambda + A_ineq' mu.
        result_.stationarity = inner.gradient_norm;
        result_.constraint_violation = violation;
        if (violation <= p.tolerance && inner.gradient_norm <= p.tolerance) {
            result_.converged = true;
            break;
        }

        if (violation > options_.required_decrease * previous_violation) {
            rho = std::min(rho * options_.penalty_growth, options_.max_penalty);
        }
        previous_violation = violation;
        inner_tol = std::max(p.tolerance, 0.1 * inner_tol);
    }

    result_.x = x;
    result_.lambda_eq = lambda;
    result_.mu_ineq = mu;
    result_.objective = objective_(x);
    result_.penalty = rho;
    return result_.x;
}
