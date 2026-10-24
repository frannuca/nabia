//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_RISKBUDGETING_H
#define NABIA_RISKBUDGETING_H
#include "nabia/optimization/IOptimizer.h"
#include "nabia/risk/IRiskMeasure.h"
#include <concepts>
#include <memory>
#include <optional>
#include <utility>
#include <Eigen/Dense>

struct RiskBudgetingParams {
    Eigen::VectorXd budgets;   // b_i > 0, normalized to sum 1; empty = equal risk contribution
    double tolerance = 1e-8;   // on max_i |RC_i / R - b_i|
    int max_iter = 5000;       // L-BFGS iterations
    int memory = 20;           // L-BFGS (s, y) pairs
};

struct RiskBudgetingResult {
    Eigen::VectorXd weights;                 // x_i > 0, sum x = 1
    double risk = 0.0;                       // R(x)
    Eigen::VectorXd risk_contributions;      // x_i dR/dx_i, sums to R(x)
    Eigen::VectorXd relative_contributions;  // risk_contributions / R(x), matches the budgets
    double budget_error = 0.0;               // max_i |relative_i - b_i|
    int iterations = 0;
    bool converged = false;
};

namespace nabia::risk::detail {
/// Non-template core shared by every RiskBudgeting<Measure>.
RiskBudgetingResult solve_risk_budgeting(const IRiskMeasure& measure, const RiskBudgetingParams& params,
                                         std::optional<Eigen::VectorXd> initial_guess);
} // namespace nabia::risk::detail

/// Long-only risk budgeting portfolio: the weights x > 0, sum x = 1, whose
/// risk contributions x_i dR/dx_i are proportional to the budgets b.
///
/// Works with any risk measure R that is convex and positively homogeneous
/// (Volatility, ExpectedShortfall, ValueAtRisk with confidence > 0.5) and
/// positive on long-only portfolios. It solves the convex problem (Roncalli, Spinu)
///     min_{y > 0}  R(y) - sum_i b_i log y_i,
/// whose optimality condition y_i dR/dy_i = b_i is exactly the budget
/// condition, and then normalizes x = y / sum(y). It is solved with LBFGS;
/// the log barrier keeps y > 0 (the line search rejects steps that leave it).
///
/// The initial guess, if given, is a vector of positive weights (any scale).
template <std::derived_from<IRiskMeasure> Measure>
class RiskBudgeting: public IOptimizer<RiskBudgetingParams, RiskBudgetingResult> {
public:
    RiskBudgeting(std::unique_ptr<RiskBudgetingParams> parameters, Measure measure)
        : IOptimizer(std::move(parameters)), measure_(std::move(measure)) {}

    /// Returns the risk budgeting weights; diagnostics are in result().
    /// Throws std::invalid_argument on invalid budgets or initial guess.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override {
        result_ = nabia::risk::detail::solve_risk_budgeting(measure_, *parameters, std::move(initial_guess));
        return result_.weights;
    }

    [[nodiscard]] const RiskBudgetingResult& result() const override { return result_; }
    [[nodiscard]] const Measure& measure() const { return measure_; }

private:
    Measure measure_;
    RiskBudgetingResult result_;
};

#endif // NABIA_RISKBUDGETING_H
