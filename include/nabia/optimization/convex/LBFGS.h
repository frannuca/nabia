//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_LBFGS_H
#define NABIA_LBFGS_H
#include "nabia/optimization/IOptimizer.h"
#include <functional>
#include <memory>
#include <Eigen/Dense>

struct LBFGSParams {
    Eigen::Index dim;
    double tolerance = 1e-8;   // stop when ||grad f||_inf <= tolerance
    int max_iter = 1000;
    int memory = 10;           // number of (s, y) pairs kept
    double armijo_c1 = 1e-4;   // sufficient decrease constant
    double backtrack = 0.5;    // step shrink factor in the line search
    int max_line_search = 60;
};

enum class LBFGSStatus {
    Converged,         // ||grad f||_inf <= tolerance
    MaxIterations,     // max_iter reached
    LineSearchFailed,  // no step satisfied the Armijo condition
};

struct LBFGSResult {
    Eigen::VectorXd x;
    double objective = 0.0;
    double gradient_norm = 0.0;  // ||grad f(x)||_inf
    int iterations = 0;
    int evaluations = 0;         // calls to the objective/gradient
    LBFGSStatus status = LBFGSStatus::MaxIterations;

    [[nodiscard]] bool converged() const { return status == LBFGSStatus::Converged; }
};

/// Limited-memory BFGS for smooth unconstrained minimization  min f(x).
/// Uses the two-loop recursion for the search direction and Armijo
/// backtracking for the step; curvature pairs with s'y <= 0 are skipped so the
/// implicit inverse Hessian stays positive definite.
class LBFGS: public IOptimizer<LBFGSParams> {
public:
    /// Returns f(x) and writes grad f(x) into `grad` (already sized to dim).
    using ValueAndGradient = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;
    using Objective = std::function<double(const Eigen::VectorXd&)>;
    using Gradient = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    LBFGS(std::unique_ptr<LBFGSParams> parameters, ValueAndGradient value_and_gradient);
    LBFGS(std::unique_ptr<LBFGSParams> parameters, Objective objective, Gradient gradient);

    /// Runs the solver from `initial_guess` (zero if empty) and returns the
    /// final point; details are in result(). Throws std::invalid_argument on bad input.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override;

    [[nodiscard]] const LBFGSResult& result() const { return result_; }

private:
    ValueAndGradient value_and_gradient_;
    LBFGSResult result_;
};

#endif // NABIA_LBFGS_H
