//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/convex/LBFGS.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

using Eigen::VectorXd;

LBFGS::LBFGS(std::unique_ptr<LBFGSParams> parameters, ValueAndGradient value_and_gradient)
    : IOptimizer(std::move(parameters)), value_and_gradient_(std::move(value_and_gradient)) {}

LBFGS::LBFGS(std::unique_ptr<LBFGSParams> parameters, Objective objective, Gradient gradient)
    : LBFGS(std::move(parameters), ValueAndGradient{}) {
    if (objective && gradient) {
        value_and_gradient_ = [f = std::move(objective), df = std::move(gradient)](const VectorXd& x, VectorXd& grad) {
            grad = df(x);
            return f(x);
        };
    }
}

Eigen::VectorXd LBFGS::optimize(std::optional<Eigen::VectorXd> initial_guess) {
    if (!parameters) {
        throw std::invalid_argument("LBFGS: parameters are null");
    }
    if (!value_and_gradient_) {
        throw std::invalid_argument("LBFGS: objective and gradient are required");
    }
    const LBFGSParams& p = *parameters;
    if (p.dim <= 0 || (initial_guess && initial_guess->size() != p.dim)) {
        throw std::invalid_argument("LBFGS: `dim` must be > 0 and the initial guess must have `dim` entries");
    }
    if (!(p.tolerance > 0.0) || p.memory < 1 || !(p.armijo_c1 > 0.0 && p.armijo_c1 < 1.0) ||
        !(p.backtrack > 0.0 && p.backtrack < 1.0)) {
        throw std::invalid_argument("LBFGS: invalid tolerance, memory or line search settings");
    }

    result_ = LBFGSResult{};
    VectorXd x = initial_guess ? std::move(*initial_guess) : VectorXd::Zero(p.dim);
    VectorXd g(x.size());
    double fx = value_and_gradient_(x, g);
    ++result_.evaluations;

    std::deque<VectorXd> s_hist;
    std::deque<VectorXd> y_hist;
    std::deque<double> rho_hist;

    int k = 0;
    for (; k < p.max_iter; ++k) {
        if (g.lpNorm<Eigen::Infinity>() <= p.tolerance) {
            result_.status = LBFGSStatus::Converged;
            break;
        }

        // Two-loop recursion: d = -H g.
        VectorXd q = g;
        std::vector<double> alpha(s_hist.size());
        for (std::size_t j = s_hist.size(); j-- > 0;) {
            alpha[j] = rho_hist[j] * s_hist[j].dot(q);
            q -= alpha[j] * y_hist[j];
        }
        const double gamma = s_hist.empty() ? 1.0 / std::max(1.0, g.norm())
                                            : s_hist.back().dot(y_hist.back()) / y_hist.back().squaredNorm();
        VectorXd d = gamma * q;
        for (std::size_t j = 0; j < s_hist.size(); ++j) {
            const double beta = rho_hist[j] * y_hist[j].dot(d);
            d += (alpha[j] - beta) * s_hist[j];
        }
        d = -d;

        double slope = g.dot(d);
        if (!(slope < 0.0)) {  // not a descent direction: restart from steepest descent
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            d = -g / std::max(1.0, g.norm());
            slope = g.dot(d);
        }

        double t = 1.0;
        VectorXd x_new;
        VectorXd g_new(x.size());
        double f_new = fx;
        bool accepted = false;
        for (int ls = 0; ls < p.max_line_search; ++ls) {
            x_new = x + t * d;
            f_new = value_and_gradient_(x_new, g_new);
            ++result_.evaluations;
            if (!std::isfinite(f_new)) {
                t *= p.backtrack;
                continue;
            }
            // Armijo sufficient decrease, or Hager-Zhang's approximate Wolfe
            // conditions when f is flat to within rounding: near the minimum
            // the decrease in f drops below machine precision, but the
            // directional derivative is still accurate.
            const double new_slope = g_new.dot(d);
            const bool armijo = f_new <= fx + p.armijo_c1 * t * slope;
            const bool approx_wolfe = f_new <= fx + p.flat_tolerance * std::abs(fx) &&
                                      new_slope <= (2.0 * p.armijo_c1 - 1.0) * slope &&
                                      new_slope >= p.wolfe_c2 * slope;
            if (armijo || approx_wolfe) {
                accepted = true;
                break;
            }
            t *= p.backtrack;
        }
        if (!accepted) {
            result_.status = LBFGSStatus::LineSearchFailed;
            break;
        }

        VectorXd s = x_new - x;
        VectorXd y = g_new - g;
        const double sy = s.dot(y);
        if (sy > 1e-12 * s.norm() * y.norm()) {  // keep H positive definite
            s_hist.push_back(std::move(s));
            y_hist.push_back(std::move(y));
            rho_hist.push_back(1.0 / sy);
            if (s_hist.size() > static_cast<std::size_t>(p.memory)) {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
        }
        x = std::move(x_new);
        g = g_new;
        fx = f_new;
    }
    // The loop can also end with k == max_iter right after a step that converged.
    if (result_.status == LBFGSStatus::MaxIterations && g.lpNorm<Eigen::Infinity>() <= p.tolerance) {
        result_.status = LBFGSStatus::Converged;
    }

    result_.x = std::move(x);
    result_.objective = fx;
    result_.gradient_norm = g.lpNorm<Eigen::Infinity>();
    result_.iterations = k;
    return result_.x;
}
