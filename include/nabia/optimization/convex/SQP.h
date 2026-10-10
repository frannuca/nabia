//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_SQP_H
#define NABIA_SQP_H

#include "nabia/optimization/convex/IConvexOptimization.h"
#include <functional>
#include <memory>
#include <optional>
#include <Eigen/Dense>

/// Tuning knobs for SQP. The defaults work for most well-scaled problems.
struct SQPOptions {
    double qp_tolerance = 1e-10;          // KKT accuracy of each QP subproblem
    int qp_max_iter = 200;                // outer iterations of the QP solver
    double merit_c1 = 1e-4;               // Armijo constant on the merit function
    double backtrack = 0.5;               // step shrink factor in the line search
    int max_line_search = 40;
    double initial_merit_penalty = 1.0;   // nu_0 in phi(x) = f(x) + nu * ||c(x)||_1
    double merit_penalty_margin = 1e-3;   // keep nu >= max |multiplier| + margin
};

struct SQPResult {
    Eigen::VectorXd x;
    Eigen::VectorXd lambda_eq;   // multipliers of A_eq x = b_eq
    Eigen::VectorXd mu_ineq;     // multipliers (>= 0) of the inequality rows (box rows as in to_linear_constraint)
    double objective = 0.0;
    double constraint_violation = 0.0;  // max |h_i| and max(g_i, 0)
    double stationarity = 0.0;          // ||grad f + A_eq' lambda + A_ineq' mu||_inf
    double complementarity = 0.0;       // max |mu_i g_i(x)|
    int iterations = 0;                 // SQP (outer) iterations
    int evaluations = 0;                // calls to objective/gradient
    bool converged = false;
};

/// Sequential Quadratic Programming for
///
///     min f(x)  s.t.  A_eq x = b_eq,  A_ineq x <= b_ineq
///
/// with f smooth. Each iteration solves the QP model
///
///     min_d  grad f(x)'d + 1/2 d'B d   s.t.  A_eq (x + d) = b_eq,  A_ineq (x + d) <= b_ineq
///
/// where B is a damped-BFGS approximation of the Hessian of the Lagrangian
/// (equal to the Hessian of f, as the constraints are linear). The step length
/// comes from a backtracking line search on the l1 merit function
/// f(x) + nu (||h(x)||_1 + ||max(g(x), 0)||_1). The QP is convex because B
/// stays positive definite, and is solved with AugmentedLagrangian.
///
/// Compared with AugmentedLagrangian, SQP needs far fewer evaluations of f,
/// at the cost of a dense n x n matrix: prefer it when f is expensive and n
/// is moderate. Constraints and parameters follow the same conventions as
/// AugmentedLagrangian. The initial guess may be infeasible.
/// See src/optimization/convex/docs/convex_optimization.tex (built as
/// convex_optimization.pdf) for the theory and the algorithm.
class SQP: public IConvexOptimization<SQPResult> {
public:
    using Objective = std::function<double(const Eigen::VectorXd&)>;
    using Gradient = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    SQP(std::unique_ptr<ConvexOptimizationParams> parameters,
        Objective objective,
        Gradient gradient,
        SQPOptions options = {});

    /// Runs the solver from `initial_guess` (zero if empty) and returns the
    /// final point; multipliers and diagnostics are in result(). Throws
    /// std::invalid_argument if dimensions or constraints are inconsistent.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override;

    [[nodiscard]] const SQPResult& result() const override { return result_; }

private:
    Objective objective_;
    Gradient gradient_;
    SQPOptions options_;
    SQPResult result_;
};

#endif // NABIA_SQP_H
