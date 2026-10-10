//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_AUGMENTEDLAGRANGIAN_H
#define NABIA_AUGMENTEDLAGRANGIAN_H

#include "nabia/optimization/convex/IConvexOptimization.h"
#include <functional>
#include <memory>
#include <Eigen/Dense>

/// Tuning knobs for the augmented Lagrangian method. The defaults work for
/// most well-scaled problems.
struct AugmentedLagrangianOptions {
    double initial_penalty = 10.0;     // rho_0
    double penalty_growth = 10.0;      // rho <- growth * rho when progress stalls
    double max_penalty = 1e10;         // upper bound for rho
    double required_decrease = 0.25;   // grow rho unless violation shrinks by this factor
    double initial_inner_tolerance = 1e-1;
    int max_inner_iter = 1000;         // L-BFGS iterations per subproblem
    int lbfgs_memory = 10;             // number of (s, y) pairs kept by L-BFGS
};

struct AugmentedLagrangianResult {
    Eigen::VectorXd x;
    Eigen::VectorXd lambda_eq;   // multipliers of A_eq x = b_eq
    Eigen::VectorXd mu_ineq;     // multipliers (>= 0) of the inequality rows, see below
    double objective = 0.0;
    double constraint_violation = 0.0;  // max |h_i| and max complementarity-aware |g_i|
    double stationarity = 0.0;          // ||grad f + A_eq' lambda + A_ineq' mu||_inf
    double penalty = 0.0;               // final rho
    int outer_iterations = 0;
    int inner_iterations = 0;
    bool converged = false;
};

/// Augmented Lagrangian (method of multipliers) for
///
///     min f(x)  s.t.  A_eq x = b_eq,  A_ineq x <= b_ineq
///
/// with f smooth and convex. Each outer iteration minimizes the augmented
/// Lagrangian in x with L-BFGS, then updates the multipliers and the penalty.
///
/// Constraints come from `ConvexOptimizationParams::constraints`:
///   - std::nullopt:     unconstrained, reduces to plain L-BFGS.
///   - LinearConstraint: used as given.
///   - BoxConstraints:   each finite bound becomes an inequality row. `mu_ineq`
///                       lists the finite upper bounds (x_i <= u_i) in index
///                       order, followed by the finite lower bounds (-x_i <= -l_i).
///                       Empty bound vectors mean "unbounded"; use +-infinity
///                       for individual unbounded coordinates.
///
/// `tolerance` is the KKT tolerance (stationarity and feasibility) and
/// `max_iter` caps the number of outer iterations.
/// See src/optimization/convex/docs/convex_optimization.tex (built as
/// convex_optimization.pdf) for the theory and the algorithm.
class AugmentedLagrangian: public IConvexOptimization {
public:
    using Objective = std::function<double(const Eigen::VectorXd&)>;
    using Gradient = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    AugmentedLagrangian(std::unique_ptr<ConvexOptimizationParams> parameters,
                      Objective objective,
                      Gradient gradient,
                      AugmentedLagrangianOptions options = {});

    /// Runs the solver from `initial_guess` (zero if empty) and returns the
    /// final point; multipliers and diagnostics are in result(). Throws
    /// std::invalid_argument if dimensions or constraints are inconsistent.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override;

    [[nodiscard]] const AugmentedLagrangianResult& result() const { return result_; }

private:
    Objective objective_;
    Gradient gradient_;
    AugmentedLagrangianOptions options_;
    AugmentedLagrangianResult result_;
};

#endif // NABIA_AUGMENTEDLAGRANGIAN_H
