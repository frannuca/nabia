//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/nongradopt/NelderMead.h"

#include "box.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

NelderMead::NelderMead(std::unique_ptr<NelderMeadParams> parameters, Objective objective)
    : IOptimizer(std::move(parameters)), objective_(std::move(objective)) {}

Eigen::VectorXd NelderMead::optimize(std::optional<Eigen::VectorXd> initial_guess) {
    if (!parameters) {
        throw std::invalid_argument("NelderMead: parameters are null");
    }
    if (!objective_) {
        throw std::invalid_argument("NelderMead: objective is required");
    }
    const NelderMeadParams& p = *parameters;
    const Index n = p.dim;
    if (n <= 0 || (initial_guess && initial_guess->size() != n)) {
        throw std::invalid_argument("NelderMead: `dim` must be > 0 and the initial guess must have `dim` entries");
    }
    if (!(p.initial_step > 0.0) || !(p.tolerance_f >= 0.0) || !(p.tolerance_x >= 0.0)) {
        throw std::invalid_argument("NelderMead: initial step must be positive and tolerances non-negative");
    }
    const nabia::nongradopt::detail::Box box = nabia::nongradopt::detail::make_box(p.bounds, n, "NelderMead");

    // Coefficients: alpha (reflection), gamma (expansion), rho (contraction), sigma (shrink).
    const double dn = static_cast<double>(n);
    const double alpha = p.adaptive ? 1.0 : p.reflection;
    const double gamma = p.adaptive ? 1.0 + 2.0 / dn : p.expansion;
    const double rho = p.adaptive ? 0.75 - 1.0 / (2.0 * dn) : p.contraction;
    const double sigma = p.adaptive ? 1.0 - 1.0 / dn : p.shrink;
    if (!(alpha > 0.0 && gamma > 1.0 && rho > 0.0 && rho < 1.0 && sigma > 0.0 && sigma < 1.0)) {
        throw std::invalid_argument("NelderMead: invalid simplex coefficients");
    }

    result_ = NelderMeadResult{};
    auto f = [&](const VectorXd& x) {
        ++result_.evaluations;
        const double v = objective_(x);
        return std::isnan(v) ? std::numeric_limits<double>::infinity() : v;
    };

    // Starting point: the guess, else the box centre where both bounds are finite, else 0.
    VectorXd x0(n);
    for (Index i = 0; i < n; ++i) {
        const double lo = box.lower(i);
        const double up = box.upper(i);
        x0(i) = std::isfinite(lo) && std::isfinite(up) ? 0.5 * (lo + up) : 0.0;
    }
    if (initial_guess) {
        x0 = std::move(*initial_guess);
    }
    x0 = box.project(x0);

    // Initial simplex: x0 and x0 + h_i e_i (or x0 - h_i e_i if that leaves the box).
    MatrixXd simplex(n, n + 1);  // vertices as columns
    simplex.col(0) = x0;
    for (Index i = 0; i < n; ++i) {
        const double width = box.upper(i) - box.lower(i);
        const double h = std::isfinite(width) ? p.initial_step * width : p.initial_step * std::max(1.0, std::abs(x0(i)));
        VectorXd v = x0;
        v(i) = x0(i) + h <= box.upper(i) ? x0(i) + h : x0(i) - h;
        simplex.col(i + 1) = box.project(v);
    }
    VectorXd values(n + 1);
    for (Index j = 0; j <= n; ++j) {
        values(j) = f(simplex.col(j));
    }

    std::vector<Index> order(static_cast<std::size_t>(n + 1));
    auto sort_simplex = [&] {
        std::iota(order.begin(), order.end(), Index{0});
        std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return values(a) < values(b); });
        MatrixXd sorted(n, n + 1);
        VectorXd sorted_values(n + 1);
        for (Index j = 0; j <= n; ++j) {
            sorted.col(j) = simplex.col(order[static_cast<std::size_t>(j)]);
            sorted_values(j) = values(order[static_cast<std::size_t>(j)]);
        }
        simplex = std::move(sorted);
        values = std::move(sorted_values);
    };

    int iter = 0;
    for (;; ++iter) {
        sort_simplex();
        // Column 0 is the best vertex, column n the worst.
        const double spread_f = values(n) - values(0);
        const double diameter = (simplex.rightCols(n).colwise() - simplex.col(0)).cwiseAbs().maxCoeff();
        if (spread_f <= p.tolerance_f * (1.0 + std::abs(values(0))) &&
            diameter <= p.tolerance_x * (1.0 + simplex.col(0).lpNorm<Eigen::Infinity>())) {
            result_.status = NelderMeadStatus::Converged;
            break;
        }
        if (iter >= p.max_iter) {
            result_.status = NelderMeadStatus::MaxIterations;
            break;
        }
        if (result_.evaluations >= p.max_evaluations) {
            result_.status = NelderMeadStatus::MaxEvaluations;
            break;
        }

        const VectorXd centroid = simplex.leftCols(n).rowwise().mean();
        const VectorXd worst = simplex.col(n);
        const VectorXd xr = box.project(centroid + alpha * (centroid - worst));
        const double fr = f(xr);

        if (fr < values(0)) {
            const VectorXd xe = box.project(centroid + gamma * (xr - centroid));
            const double fe = f(xe);
            if (fe < fr) {
                simplex.col(n) = xe;
                values(n) = fe;
            } else {
                simplex.col(n) = xr;
                values(n) = fr;
            }
            continue;
        }
        if (fr < values(n - 1)) {
            simplex.col(n) = xr;
            values(n) = fr;
            continue;
        }
        // Contraction: outside if the reflection beat the worst vertex, inside otherwise.
        const bool outside = fr < values(n);
        const VectorXd xc = outside ? VectorXd(box.project(centroid + rho * (xr - centroid)))
                                    : VectorXd(box.project(centroid + rho * (worst - centroid)));
        const double fc = f(xc);
        if (fc < (outside ? fr : values(n))) {
            simplex.col(n) = xc;
            values(n) = fc;
            continue;
        }
        // Shrink towards the best vertex (stays inside the box: the box is convex).
        for (Index j = 1; j <= n; ++j) {
            simplex.col(j) = simplex.col(0) + sigma * (simplex.col(j) - simplex.col(0));
            values(j) = f(simplex.col(j));
        }
    }

    result_.x = simplex.col(0);
    result_.objective = values(0);
    result_.iterations = iter;
    return result_.x;
}
