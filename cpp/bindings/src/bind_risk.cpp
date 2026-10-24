#include "bridge.hpp"

#include <nabia/risk/IRiskMeasure.h>
#include <nabia/risk/RiskBudgeting.h>
#include <nabia/risk/measures/ExpectedShortfall.h>
#include <nabia/risk/measures/ValueAtRisk.h>
#include <nabia/risk/measures/Volatility.h>

#include <optional>
#include <variant>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace nabia_bridge {

namespace {

/// RiskBudgeting<Measure> for whichever measure Python passes: the C++ class
/// is a template over the measure type.
class AnyRiskBudgeting {
public:
    AnyRiskBudgeting(const RiskBudgetingParams& params, const IRiskMeasure& measure) {
        if (const auto* v = dynamic_cast<const Volatility*>(&measure)) {
            impl_ = make(params, *v);
        } else if (const auto* var = dynamic_cast<const ValueAtRisk*>(&measure)) {
            impl_ = make(params, *var);
        } else if (const auto* es = dynamic_cast<const ExpectedShortfall*>(&measure)) {
            impl_ = make(params, *es);
        } else {
            throw std::invalid_argument("RiskBudgeting: unsupported risk measure");
        }
    }

    VectorXd optimize(std::optional<VectorXd> x0) {
        return std::visit([&](auto& rb) { return rb->optimize(std::move(x0)); }, impl_);
    }
    [[nodiscard]] RiskBudgetingResult result() const {
        return std::visit([](const auto& rb) { return rb->result(); }, impl_);
    }
    [[nodiscard]] RiskBudgetingParams parameters() const { return params_; }
    [[nodiscard]] const IRiskMeasure& measure() const {
        return std::visit([](const auto& rb) -> const IRiskMeasure& { return rb->measure(); }, impl_);
    }

private:
    template <typename Measure>
    std::unique_ptr<RiskBudgeting<Measure>> make(const RiskBudgetingParams& params, const Measure& measure) {
        params_ = params;
        return std::make_unique<RiskBudgeting<Measure>>(std::make_unique<RiskBudgetingParams>(params), measure);
    }

    RiskBudgetingParams params_;
    std::variant<std::unique_ptr<RiskBudgeting<Volatility>>, std::unique_ptr<RiskBudgeting<ValueAtRisk>>,
                 std::unique_ptr<RiskBudgeting<ExpectedShortfall>>>
        impl_;
};

} // namespace

void bind_risk(py::module_& m) {
    py::class_<IRiskMeasure>(m, "IRiskMeasure", "Risk of a portfolio x from an asset covariance matrix.")
        .def("compute_risk", &IRiskMeasure::compute_risk, py::arg("x"))
        .def("compute_risk_contributions", &IRiskMeasure::compute_risk_contributions, py::arg("x"),
             "x_i dR/dx_i; they sum to R(x).")
        .def("compute_marginal_risk_contributions", &IRiskMeasure::compute_marginal_risk_contributions, py::arg("x"),
             "dR/dx_i.")
        .def_property_readonly("dim", &IRiskMeasure::dim)
        .def_property_readonly("covariance_matrix", &IRiskMeasure::covariance_matrix);
    py::class_<Volatility, IRiskMeasure>(m, "Volatility", "sigma(x) = sqrt(x' Sigma x).")
        .def(py::init<MatrixXd>(), py::arg("covariance_matrix"));
    py::class_<ValueAtRisk, IRiskMeasure>(m, "ValueAtRisk", "Gaussian VaR: -mu'x + z_alpha sigma(x).")
        .def(py::init<MatrixXd, double, VectorXd>(), py::arg("covariance_matrix"), py::arg("confidence_level"),
             py::arg("expected_returns") = VectorXd())
        .def_property_readonly("confidence_level", &ValueAtRisk::confidence_level);
    py::class_<ExpectedShortfall, IRiskMeasure>(m, "ExpectedShortfall",
                                                "Gaussian ES: -mu'x + phi(z_alpha) / (1 - alpha) sigma(x).")
        .def(py::init<MatrixXd, double, VectorXd>(), py::arg("covariance_matrix"), py::arg("confidence_level"),
             py::arg("expected_returns") = VectorXd())
        .def_property_readonly("confidence_level", &ExpectedShortfall::confidence_level);

    py::class_<RiskBudgetingParams>(m, "RiskBudgetingParams")
        .def(kwargs_init<RiskBudgetingParams>())
        .def_readwrite("budgets", &RiskBudgetingParams::budgets, "b_i > 0 (normalized); empty = equal risk contribution.")
        .def_readwrite("tolerance", &RiskBudgetingParams::tolerance)
        .def_readwrite("max_iter", &RiskBudgetingParams::max_iter)
        .def_readwrite("memory", &RiskBudgetingParams::memory);
    py::class_<RiskBudgetingResult>(m, "RiskBudgetingResult")
        .def_readonly("weights", &RiskBudgetingResult::weights)
        .def_readonly("risk", &RiskBudgetingResult::risk)
        .def_readonly("risk_contributions", &RiskBudgetingResult::risk_contributions)
        .def_readonly("relative_contributions", &RiskBudgetingResult::relative_contributions)
        .def_readonly("budget_error", &RiskBudgetingResult::budget_error)
        .def_readonly("iterations", &RiskBudgetingResult::iterations)
        .def_readonly("converged", &RiskBudgetingResult::converged);
    py::class_<AnyRiskBudgeting>(m, "RiskBudgeting",
                                 "Long-only weights (sum 1) whose risk contributions match the budgets.")
        .def(py::init<const RiskBudgetingParams&, const IRiskMeasure&>(), py::arg("parameters"), py::arg("measure"))
        .def(
            "optimize",
            [](AnyRiskBudgeting& rb, std::optional<VectorXd> x0) {
                const py::gil_scoped_release release;
                return rb.optimize(std::move(x0));
            },
            py::arg("initial_guess") = py::none())
        .def_property_readonly("result", &AnyRiskBudgeting::result)
        .def_property_readonly("parameters", &AnyRiskBudgeting::parameters)
        .def_property_readonly("measure", &AnyRiskBudgeting::measure, py::return_value_policy::reference_internal);
}

} // namespace nabia_bridge
