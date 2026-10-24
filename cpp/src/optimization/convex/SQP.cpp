//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/convex/SQP.h"
#include "nabia/optimization/convex/AugmentedLagrangian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct QPSolution {
    VectorXd d;
    VectorXd lambda;
    VectorXd mu;
};

/// min_d g'd + 1/2 d'B d  s.t.  A_eq d = r_eq,  A_ineq d <= r_ineq  (B positive definite).
QPSolution solve_qp(const MatrixXd& B, const VectorXd& g, const LinearConstraint& shifted, const SQPOptions& options) {
    const Index n = g.size();
    if (shifted.num_eq() == 0 && shifted.num_ineq() == 0) {
        return {B.llt().solve(-g), VectorXd(0), VectorXd(0)};
    }
    auto params = std::make_unique<ConvexOptimizationParams>(
        ConvexOptimizationParams{n, options.qp_tolerance, static_cast<double>(options.qp_max_iter), shifted});
    AugmentedLagrangian qp(
        std::move(params), [&](const VectorXd& d) { return g.dot(d) + 0.5 * d.dot(B * d); },
        [&](const VectorXd& d) -> VectorXd { return g + B * d; });
    VectorXd d = qp.optimize(VectorXd::Zero(n));
    return {std::move(d), qp.result().lambda_eq, qp.result().mu_ineq};
}

/// Damped BFGS update (Powell): keeps B positive definite even when s'y <= 0.
void damped_bfgs_update(MatrixXd& B, const VectorXd& s, const VectorXd& y) {
    const VectorXd Bs = B * s;
    const double sBs = s.dot(Bs);
    if (!(sBs > 0.0)) {
        return;
    }
    const double sy = s.dot(y);
    const double theta = sy >= 0.2 * sBs ? 1.0 : 0.8 * sBs / (sBs - sy);
    const VectorXd r = theta * y + (1.0 - theta) * Bs;
    B.noalias() -= Bs * Bs.transpose() / sBs;
    B.noalias() += r * r.transpose() / s.dot(r);
    B = 0.5 * (B + B.transpose());  // remove round-off asymmetry
}

} // namespace

SQP::SQP(std::unique_ptr<ConvexOptimizationParams> parameters,
         Objective objective,
         Gradient gradient,
         SQPOptions options)
    : IConvexOptimization(std::move(parameters)),
      objective_(std::move(objective)),
      gradient_(std::move(gradient)),
      options_(options) {}

Eigen::VectorXd SQP::optimize(std::optional<Eigen::VectorXd> initial_guess) {
    if (!parameters) {
        throw std::invalid_argument("SQP: parameters are null");
    }
    if (!objective_ || !gradient_) {
        throw std::invalid_argument("SQP: objective and gradient are required");
    }
    const ConvexOptimizationParams& p = *parameters;
    const Index n = p.dim;
    if (n <= 0 || (initial_guess && initial_guess->size() != n)) {
        throw std::invalid_argument(
            "SQP: `dim` must be > 0 and the initial guess must have `dim` entries");
    }
    if (!(p.tolerance > 0.0) || !(options_.backtrack > 0.0 && options_.backtrack < 1.0)) {
        throw std::invalid_argument("SQP: tolerance and backtrack must be in range");
    }

    const LinearConstraint lc = to_linear_constraint(p.constraints, n);

    // h = A_eq x - b_eq, g = A_ineq x - b_ineq
    auto l1_violation = [&](const VectorXd& z) {
        double v = 0.0;
        if (lc.num_eq() > 0) {
            v += (lc.A_eq * z - lc.b_eq).lpNorm<1>();
        }
        if (lc.num_ineq() > 0) {
            v += (lc.A_ineq * z - lc.b_ineq).cwiseMax(0.0).sum();
        }
        return v;
    };
    auto max_violation = [&](const VectorXd& z) {
        double v = 0.0;
        if (lc.num_eq() > 0) {
            v = (lc.A_eq * z - lc.b_eq).lpNorm<Eigen::Infinity>();
        }
        if (lc.num_ineq() > 0) {
            v = std::max(v, (lc.A_ineq * z - lc.b_ineq).cwiseMax(0.0).maxCoeff());
        }
        return v;
    };

    result_ = SQPResult{};
    VectorXd x = initial_guess ? std::move(*initial_guess) : VectorXd::Zero(n);
    double fx = objective_(x);
    VectorXd gx = gradient_(x);
    result_.evaluations = 1;
    if (!std::isfinite(fx) || !gx.allFinite()) {
        throw std::invalid_argument("SQP: objective is not finite at the initial guess");
    }

    MatrixXd B = MatrixXd::Identity(n, n);
    bool scaled = false;  // B = (y'y / s'y) I after the first step
    double nu = options_.initial_merit_penalty;
    VectorXd lambda = VectorXd::Zero(lc.num_eq());
    VectorXd mu = VectorXd::Zero(lc.num_ineq());
    const int max_outer = static_cast<int>(p.max_iter);

    for (int k = 0; k < max_outer; ++k) {
        LinearConstraint shifted = lc;
        if (lc.num_eq() > 0) {
            shifted.b_eq = lc.b_eq - lc.A_eq * x;
        }
        if (lc.num_ineq() > 0) {
            shifted.b_ineq = lc.b_ineq - lc.A_ineq * x;
        }
        QPSolution qp = solve_qp(B, gx, shifted, options_);
        lambda = std::move(qp.lambda);
        mu = std::move(qp.mu);
        const VectorXd& d = qp.d;

        // KKT residuals at x with the QP multipliers.
        VectorXd grad_lagrangian = gx;
        double complementarity = 0.0;
        if (lc.num_eq() > 0) {
            grad_lagrangian.noalias() += lc.A_eq.transpose() * lambda;
        }
        if (lc.num_ineq() > 0) {
            grad_lagrangian.noalias() += lc.A_ineq.transpose() * mu;
            complementarity = mu.cwiseProduct(lc.A_ineq * x - lc.b_ineq).lpNorm<Eigen::Infinity>();
        }
        result_.iterations = k;
        result_.stationarity = grad_lagrangian.lpNorm<Eigen::Infinity>();
        result_.constraint_violation = max_violation(x);
        result_.complementarity = complementarity;
        if (result_.stationarity <= p.tolerance && result_.constraint_violation <= p.tolerance &&
            complementarity <= p.tolerance) {
            result_.converged = true;
            break;
        }

        // l1 merit function phi = f + nu * V1; nu must exceed the multipliers
        // for d to be a descent direction of phi.
        double max_multiplier = 0.0;
        if (lambda.size() > 0) {
            max_multiplier = lambda.lpNorm<Eigen::Infinity>();
        }
        if (mu.size() > 0) {
            max_multiplier = std::max(max_multiplier, mu.lpNorm<Eigen::Infinity>());
        }
        nu = std::max(nu, max_multiplier + options_.merit_penalty_margin);
        const double v1 = l1_violation(x);
        const double merit = fx + nu * v1;
        // Directional derivative of phi along d (the constraints are linear and
        // d satisfies them, so V1(x + t d) <= (1 - t) V1(x)).
        double slope = gx.dot(d) - nu * v1;
        if (!(slope < 0.0)) {
            slope = -1e-16 * std::max(1.0, std::abs(merit));  // d ~ 0: accept the first decrease
        }

        double t = 1.0;
        VectorXd x_new;
        double f_new = fx;
        bool accepted = false;
        for (int ls = 0; ls < options_.max_line_search; ++ls) {
            x_new = x + t * d;
            f_new = objective_(x_new);
            ++result_.evaluations;
            if (std::isfinite(f_new) && f_new + nu * l1_violation(x_new) <= merit + options_.merit_c1 * t * slope) {
                accepted = true;
                break;
            }
            t *= options_.backtrack;
        }
        if (!accepted) {
            if (B.isIdentity()) {
                break;  // no progress even along the steepest-descent-like QP step
            }
            B.setIdentity();  // restart the quasi-Newton model
            scaled = false;
            continue;
        }

        VectorXd g_new = gradient_(x_new);
        const VectorXd s = x_new - x;
        const VectorXd y = g_new - gx;
        if (!scaled && s.dot(y) > 0.0) {
            B = (y.squaredNorm() / s.dot(y)) * MatrixXd::Identity(n, n);
            scaled = true;
        }
        damped_bfgs_update(B, s, y);

        x = std::move(x_new);
        fx = f_new;
        gx = std::move(g_new);
        result_.iterations = k + 1;
    }

    result_.x = x;
    result_.lambda_eq = lambda;
    result_.mu_ineq = mu;
    result_.objective = fx;
    return result_.x;
}
