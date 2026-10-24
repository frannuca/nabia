//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/convex/Constraints.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <variant>

namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double kInf = std::numeric_limits<double>::infinity();

LinearConstraint unconstrained(Index n) {
    return {MatrixXd(0, n), VectorXd(0), MatrixXd(0, n), VectorXd(0)};
}

LinearConstraint box_to_linear_constraint(const BoxConstraints& box, Index n) {
    const auto size = static_cast<std::size_t>(n);
    const bool has_lower = !box.lower_bounds.empty();
    const bool has_upper = !box.upper_bounds.empty();
    if ((has_lower && box.lower_bounds.size() != size) || (has_upper && box.upper_bounds.size() != size)) {
        throw std::invalid_argument("Constraints: box bounds must be empty or have `dim` entries");
    }

    struct Row {
        Index i;
        double sign;
        double rhs;
    };
    std::vector<Row> rows;
    for (std::size_t j = 0; j < size; ++j) {
        const double lo = has_lower ? box.lower_bounds[j] : -kInf;
        const double up = has_upper ? box.upper_bounds[j] : kInf;
        if (std::isnan(lo) || std::isnan(up) || lo > up) {
            throw std::invalid_argument("Constraints: box bounds need lower <= upper");
        }
    }
    for (std::size_t j = 0; has_upper && j < size; ++j) {
        if (std::isfinite(box.upper_bounds[j])) {
            rows.push_back({static_cast<Index>(j), 1.0, box.upper_bounds[j]});
        }
    }
    for (std::size_t j = 0; has_lower && j < size; ++j) {
        if (std::isfinite(box.lower_bounds[j])) {
            rows.push_back({static_cast<Index>(j), -1.0, -box.lower_bounds[j]});
        }
    }

    const auto m = static_cast<Index>(rows.size());
    LinearConstraint lf{MatrixXd(0, n), VectorXd(0), MatrixXd::Zero(m, n), VectorXd(m)};
    for (Index r = 0; r < m; ++r) {
        const Row& row = rows[static_cast<std::size_t>(r)];
        lf.A_ineq(r, row.i) = row.sign;
        lf.b_ineq(r) = row.rhs;
    }
    return lf;
}

} // namespace

LinearConstraint to_linear_constraint(const std::optional<Constraints>& constraints, Eigen::Index n) {
    if (!constraints) {
        return unconstrained(n);
    }
    if (const auto* box = std::get_if<BoxConstraints>(&*constraints)) {
        return box_to_linear_constraint(*box, n);
    }
    const auto& lin = std::get<LinearConstraint>(*constraints);
    if (!lin.is_consistent(n)) {
        throw std::invalid_argument("Constraints: linear constraint dimensions do not match `dim`");
    }
    LinearConstraint lf = unconstrained(n);
    if (lin.num_eq() > 0) {
        lf.A_eq = lin.A_eq;
        lf.b_eq = lin.b_eq;
    }
    if (lin.num_ineq() > 0) {
        lf.A_ineq = lin.A_ineq;
        lf.b_ineq = lin.b_ineq;
    }
    return lf;
}

