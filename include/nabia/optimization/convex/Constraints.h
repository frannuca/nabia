//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_CONVEX_CONSTRAINTS_H
#define NABIA_CONVEX_CONSTRAINTS_H
#include <optional>
#include <variant>
#include <vector>
#include <Eigen/Dense>

struct BoxConstraints{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

/// Linear constraints in matrix form:
///   A_eq   * x == b_eq     (m_eq   x n)
///   A_ineq * x <= b_ineq   (m_ineq x n)
/// Leave a pair empty (0 rows) when that kind of constraint is not used.
/// For `A x >= b`, pass `-A x <= -b`.
struct LinearConstraint {
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
    Eigen::MatrixXd A_ineq;
    Eigen::VectorXd b_ineq;

    [[nodiscard]] Eigen::Index num_eq() const { return A_eq.rows(); }
    [[nodiscard]] Eigen::Index num_ineq() const { return A_ineq.rows(); }

    /// True if all matrices have `dim` columns and each b matches its A's rows.
    [[nodiscard]] bool is_consistent(Eigen::Index dim) const {
        return (A_eq.rows() == 0 || A_eq.cols() == dim) && A_eq.rows() == b_eq.size() &&
               (A_ineq.rows() == 0 || A_ineq.cols() == dim) && A_ineq.rows() == b_ineq.size();
    }

    /// True if `x` satisfies every constraint within `tol`.
    [[nodiscard]] bool is_satisfied(const Eigen::VectorXd& x, double tol = 1e-9) const {
        if (num_eq() > 0 && ((A_eq * x - b_eq).cwiseAbs().array() > tol).any()) {
            return false;
        }
        if (num_ineq() > 0 && ((A_ineq * x - b_ineq).array() > tol).any()) {
            return false;
        }
        return true;
    }
};

/// Either box bounds or general linear constraints.
using Constraints = std::variant<BoxConstraints, LinearConstraint>;

/// Matrix form of any constraint description, with `dim` columns (zero rows
/// when unconstrained). Box bounds become inequality rows: the finite upper
/// bounds x_i <= u_i in index order, then the finite lower bounds -x_i <= -l_i.
/// Empty bound vectors mean unbounded; use +-infinity per coordinate.
/// Throws std::invalid_argument on mismatched dimensions or lower > upper.
[[nodiscard]] LinearConstraint to_linear_constraint(const std::optional<Constraints>& constraints, Eigen::Index dim);

#endif // NABIA_CONVEX_CONSTRAINTS_H
