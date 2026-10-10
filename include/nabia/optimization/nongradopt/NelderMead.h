//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_NELDERMEAD_H
#define NABIA_NELDERMEAD_H

#include "nabia/optimization/IOptimizer.h"
#include "nabia/optimization/convex/Constraints.h"
#include <functional>
#include <memory>
#include <optional>
#include <Eigen/Dense>

struct NelderMeadParams {
    Eigen::Index dim;
    std::optional<BoxConstraints> bounds;  // empty = unconstrained; +-infinity allowed per coordinate
    double tolerance_f = 1e-10;   // stop when max f - min f over the simplex <= tolerance_f (1 + |f_best|)
    double tolerance_x = 1e-10;   // ... and the simplex diameter (inf-norm) <= tolerance_x (1 + |x_best|_inf)
    int max_iter = 20000;
    int max_evaluations = 40000;
    double initial_step = 0.05;   // initial simplex edge: initial_step * (u - l) when bounded, else * max(1, |x0_i|)
    bool adaptive = true;         // dimension-dependent coefficients (Gao & Han 2012), better for dim > 2
    // Classic coefficients, used when adaptive == false.
    double reflection = 1.0;
    double expansion = 2.0;
    double contraction = 0.5;
    double shrink = 0.5;
};

enum class NelderMeadStatus {
    Converged,           // both tolerances met
    MaxIterations,
    MaxEvaluations,
};

struct NelderMeadResult {
    Eigen::VectorXd x;
    double objective = 0.0;
    int iterations = 0;
    int evaluations = 0;
    NelderMeadStatus status = NelderMeadStatus::MaxIterations;

    [[nodiscard]] bool converged() const { return status == NelderMeadStatus::Converged; }
};

/// Nelder-Mead downhill simplex for  min f(x)  s.t.  l <= x <= u.
///
/// Derivative free: only f is evaluated. Each iteration replaces the worst
/// vertex of a simplex of dim + 1 points by its reflection, expansion or
/// contraction through the centroid of the others, or shrinks the simplex
/// towards the best vertex. Box constraints are enforced by projecting every
/// trial point onto the box, so f is never evaluated outside it.
///
/// Nelder-Mead is a local method and can stall on a degenerate simplex; for
/// multimodal problems use DifferentialEvolution (possibly followed by
/// Nelder-Mead to refine the result).
class NelderMead: public IOptimizer<NelderMeadParams, NelderMeadResult> {
public:
    using Objective = std::function<double(const Eigen::VectorXd&)>;

    NelderMead(std::unique_ptr<NelderMeadParams> parameters, Objective objective);

    /// Starts from `initial_guess` (projected onto the box), or from the box
    /// centre / zero if empty. Returns the best vertex; details in result().
    /// Throws std::invalid_argument on bad input.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override;

    [[nodiscard]] const NelderMeadResult& result() const override { return result_; }

private:
    Objective objective_;
    NelderMeadResult result_;
};

#endif // NABIA_NELDERMEAD_H
