#include "bridge.hpp"

#include <nabia/optimization/convex/AugmentedLagrangian.h>
#include <nabia/optimization/convex/Constraints.h>
#include <nabia/optimization/convex/IConvexOptimization.h>
#include <nabia/optimization/convex/LBFGS.h>
#include <nabia/optimization/convex/SQP.h>
#include <nabia/optimization/nongradopt/DifferentialEvolution.h>
#include <nabia/optimization/nongradopt/NelderMead.h>

#include <optional>
#include <string>

using Eigen::Index;
using Eigen::VectorXd;

namespace nabia_bridge {

namespace {

using Objective = std::function<double(const VectorXd&)>;
using Gradient = std::function<VectorXd(const VectorXd&)>;

Objective objective(py::function f) {
    return [c = PyCallable(std::move(f))](const VectorXd& x) { return c.call<double>(x); };
}

Gradient gradient(py::function f) {
    return [c = PyCallable(std::move(f))](const VectorXd& x) {
        VectorXd g = c.call<VectorXd>(x);
        if (g.size() != x.size()) {
            throw std::invalid_argument("gradient returned " + std::to_string(g.size()) + " entries for a point of " +
                                        std::to_string(x.size()));
        }
        return g;
    };
}

/// f(x) -> (value, gradient), for LBFGS.
LBFGS::ValueAndGradient value_and_gradient(py::function f) {
    return [c = PyCallable(std::move(f))](const VectorXd& x, VectorXd& grad) {
        const auto [value, g] = c.call<std::pair<double, VectorXd>>(x);
        if (g.size() != x.size()) {
            throw std::invalid_argument("value_and_gradient returned a gradient of " + std::to_string(g.size()) +
                                        " entries for a point of " + std::to_string(x.size()));
        }
        grad = g;
        return value;
    };
}

/// optimize(initial_guess=None) with the GIL released (callbacks take it back),
/// and a `result` property returning a copy of the last result.
template <typename Solver>
void bind_solver(py::class_<Solver>& c) {
    c.def(
         "optimize",
         [](Solver& s, std::optional<VectorXd> x0) {
             const py::gil_scoped_release release;
             return s.optimize(std::move(x0));
         },
         py::arg("initial_guess") = py::none(),
         "Runs the solver and returns the solution; details are in `result`.")
        .def_property_readonly("result", [](const Solver& s) { return s.result(); });
}

} // namespace

void bind_optimization(py::module_& m) {
    // --- Constraints ---
    py::class_<BoxConstraints>(m, "BoxConstraints", "Per-coordinate bounds l <= x <= u (+-inf allowed where noted).")
        .def(py::init<>())
        .def(py::init([](std::vector<double> lower, std::vector<double> upper) {
                 return BoxConstraints{std::move(lower), std::move(upper)};
             }),
             py::arg("lower_bounds"), py::arg("upper_bounds"))
        .def_readwrite("lower_bounds", &BoxConstraints::lower_bounds)
        .def_readwrite("upper_bounds", &BoxConstraints::upper_bounds);
    py::class_<LinearConstraint>(m, "LinearConstraint", "A_eq x = b_eq and A_ineq x <= b_ineq.")
        .def(kwargs_init<LinearConstraint>())
        .def_readwrite("A_eq", &LinearConstraint::A_eq)
        .def_readwrite("b_eq", &LinearConstraint::b_eq)
        .def_readwrite("A_ineq", &LinearConstraint::A_ineq)
        .def_readwrite("b_ineq", &LinearConstraint::b_ineq)
        .def_property_readonly("num_eq", &LinearConstraint::num_eq)
        .def_property_readonly("num_ineq", &LinearConstraint::num_ineq)
        .def("is_consistent", &LinearConstraint::is_consistent, py::arg("dim"))
        .def("is_satisfied", &LinearConstraint::is_satisfied, py::arg("x"), py::arg("tol") = 1e-9);
    m.def("to_linear_constraint", &to_linear_constraint, py::arg("constraints"), py::arg("dim"),
          "Box or linear constraints (or None) as one LinearConstraint.");

    py::class_<ConvexOptimizationParams>(m, "ConvexOptimizationParams")
        .def(kwargs_init<ConvexOptimizationParams>())
        .def_readwrite("dim", &ConvexOptimizationParams::dim)
        .def_readwrite("tolerance", &ConvexOptimizationParams::tolerance)
        .def_readwrite("max_iter", &ConvexOptimizationParams::max_iter)
        .def_readwrite("constraints", &ConvexOptimizationParams::constraints,
                       "BoxConstraints, LinearConstraint or None (unconstrained).");

    // --- L-BFGS ---
    py::enum_<LBFGSStatus>(m, "LBFGSStatus")
        .value("Converged", LBFGSStatus::Converged)
        .value("MaxIterations", LBFGSStatus::MaxIterations)
        .value("LineSearchFailed", LBFGSStatus::LineSearchFailed);
    py::class_<LBFGSParams>(m, "LBFGSParams")
        .def(kwargs_init<LBFGSParams>())
        .def_readwrite("dim", &LBFGSParams::dim)
        .def_readwrite("tolerance", &LBFGSParams::tolerance)
        .def_readwrite("max_iter", &LBFGSParams::max_iter)
        .def_readwrite("memory", &LBFGSParams::memory)
        .def_readwrite("armijo_c1", &LBFGSParams::armijo_c1)
        .def_readwrite("backtrack", &LBFGSParams::backtrack)
        .def_readwrite("max_line_search", &LBFGSParams::max_line_search)
        .def_readwrite("flat_tolerance", &LBFGSParams::flat_tolerance)
        .def_readwrite("wolfe_c2", &LBFGSParams::wolfe_c2);
    py::class_<LBFGSResult>(m, "LBFGSResult")
        .def_readonly("x", &LBFGSResult::x)
        .def_readonly("objective", &LBFGSResult::objective)
        .def_readonly("gradient_norm", &LBFGSResult::gradient_norm)
        .def_readonly("iterations", &LBFGSResult::iterations)
        .def_readonly("evaluations", &LBFGSResult::evaluations)
        .def_readonly("status", &LBFGSResult::status)
        .def_property_readonly("converged", &LBFGSResult::converged);
    py::class_<LBFGS> lbfgs(m, "LBFGS", "Unconstrained limited-memory BFGS.");
    lbfgs
        .def(py::init([](const LBFGSParams& p, py::function f, py::function g) {
                 return std::make_unique<LBFGS>(std::make_unique<LBFGSParams>(p), objective(std::move(f)),
                                                gradient(std::move(g)));
             }),
             py::arg("parameters"), py::arg("objective"), py::arg("gradient"),
             "objective(x) -> float and gradient(x) -> array.")
        .def(py::init([](const LBFGSParams& p, py::function fg) {
                 return std::make_unique<LBFGS>(std::make_unique<LBFGSParams>(p), value_and_gradient(std::move(fg)));
             }),
             py::arg("parameters"), py::arg("value_and_gradient"), "value_and_gradient(x) -> (float, array).");
    bind_solver(lbfgs);

    // --- Augmented Lagrangian ---
    py::class_<AugmentedLagrangianOptions>(m, "AugmentedLagrangianOptions")
        .def(kwargs_init<AugmentedLagrangianOptions>())
        .def_readwrite("initial_penalty", &AugmentedLagrangianOptions::initial_penalty)
        .def_readwrite("penalty_growth", &AugmentedLagrangianOptions::penalty_growth)
        .def_readwrite("max_penalty", &AugmentedLagrangianOptions::max_penalty)
        .def_readwrite("required_decrease", &AugmentedLagrangianOptions::required_decrease)
        .def_readwrite("initial_inner_tolerance", &AugmentedLagrangianOptions::initial_inner_tolerance)
        .def_readwrite("max_inner_iter", &AugmentedLagrangianOptions::max_inner_iter)
        .def_readwrite("lbfgs_memory", &AugmentedLagrangianOptions::lbfgs_memory);
    py::class_<AugmentedLagrangianResult>(m, "AugmentedLagrangianResult")
        .def_readonly("x", &AugmentedLagrangianResult::x)
        .def_readonly("lambda_eq", &AugmentedLagrangianResult::lambda_eq)
        .def_readonly("mu_ineq", &AugmentedLagrangianResult::mu_ineq)
        .def_readonly("objective", &AugmentedLagrangianResult::objective)
        .def_readonly("constraint_violation", &AugmentedLagrangianResult::constraint_violation)
        .def_readonly("stationarity", &AugmentedLagrangianResult::stationarity)
        .def_readonly("penalty", &AugmentedLagrangianResult::penalty)
        .def_readonly("outer_iterations", &AugmentedLagrangianResult::outer_iterations)
        .def_readonly("inner_iterations", &AugmentedLagrangianResult::inner_iterations)
        .def_readonly("converged", &AugmentedLagrangianResult::converged);
    py::class_<AugmentedLagrangian> al(m, "AugmentedLagrangian", "Smooth objective with linear and box constraints.");
    al.def(py::init([](const ConvexOptimizationParams& p, py::function f, py::function g,
                       const AugmentedLagrangianOptions& options) {
               return std::make_unique<AugmentedLagrangian>(std::make_unique<ConvexOptimizationParams>(p),
                                                            objective(std::move(f)), gradient(std::move(g)), options);
           }),
           py::arg("parameters"), py::arg("objective"), py::arg("gradient"),
           py::arg("options") = AugmentedLagrangianOptions{});
    bind_solver(al);

    // --- SQP ---
    py::class_<SQPOptions>(m, "SQPOptions")
        .def(kwargs_init<SQPOptions>())
        .def_readwrite("qp_tolerance", &SQPOptions::qp_tolerance)
        .def_readwrite("qp_max_iter", &SQPOptions::qp_max_iter)
        .def_readwrite("merit_c1", &SQPOptions::merit_c1)
        .def_readwrite("backtrack", &SQPOptions::backtrack)
        .def_readwrite("max_line_search", &SQPOptions::max_line_search)
        .def_readwrite("initial_merit_penalty", &SQPOptions::initial_merit_penalty)
        .def_readwrite("merit_penalty_margin", &SQPOptions::merit_penalty_margin);
    py::class_<SQPResult>(m, "SQPResult")
        .def_readonly("x", &SQPResult::x)
        .def_readonly("lambda_eq", &SQPResult::lambda_eq)
        .def_readonly("mu_ineq", &SQPResult::mu_ineq)
        .def_readonly("objective", &SQPResult::objective)
        .def_readonly("constraint_violation", &SQPResult::constraint_violation)
        .def_readonly("stationarity", &SQPResult::stationarity)
        .def_readonly("complementarity", &SQPResult::complementarity)
        .def_readonly("iterations", &SQPResult::iterations)
        .def_readonly("evaluations", &SQPResult::evaluations)
        .def_readonly("converged", &SQPResult::converged);
    py::class_<SQP> sqp(m, "SQP", "Sequential quadratic programming with linear and box constraints.");
    sqp.def(py::init([](const ConvexOptimizationParams& p, py::function f, py::function g, const SQPOptions& options) {
                return std::make_unique<SQP>(std::make_unique<ConvexOptimizationParams>(p), objective(std::move(f)),
                                             gradient(std::move(g)), options);
            }),
            py::arg("parameters"), py::arg("objective"), py::arg("gradient"), py::arg("options") = SQPOptions{});
    bind_solver(sqp);

    // --- Nelder-Mead ---
    py::enum_<NelderMeadStatus>(m, "NelderMeadStatus")
        .value("Converged", NelderMeadStatus::Converged)
        .value("MaxIterations", NelderMeadStatus::MaxIterations)
        .value("MaxEvaluations", NelderMeadStatus::MaxEvaluations);
    py::class_<NelderMeadParams>(m, "NelderMeadParams")
        .def(kwargs_init<NelderMeadParams>())
        .def_readwrite("dim", &NelderMeadParams::dim)
        .def_readwrite("bounds", &NelderMeadParams::bounds, "BoxConstraints or None.")
        .def_readwrite("tolerance_f", &NelderMeadParams::tolerance_f)
        .def_readwrite("tolerance_x", &NelderMeadParams::tolerance_x)
        .def_readwrite("max_iter", &NelderMeadParams::max_iter)
        .def_readwrite("max_evaluations", &NelderMeadParams::max_evaluations)
        .def_readwrite("initial_step", &NelderMeadParams::initial_step)
        .def_readwrite("adaptive", &NelderMeadParams::adaptive)
        .def_readwrite("reflection", &NelderMeadParams::reflection)
        .def_readwrite("expansion", &NelderMeadParams::expansion)
        .def_readwrite("contraction", &NelderMeadParams::contraction)
        .def_readwrite("shrink", &NelderMeadParams::shrink);
    py::class_<NelderMeadResult>(m, "NelderMeadResult")
        .def_readonly("x", &NelderMeadResult::x)
        .def_readonly("objective", &NelderMeadResult::objective)
        .def_readonly("iterations", &NelderMeadResult::iterations)
        .def_readonly("evaluations", &NelderMeadResult::evaluations)
        .def_readonly("status", &NelderMeadResult::status)
        .def_property_readonly("converged", &NelderMeadResult::converged);
    py::class_<NelderMead> nm(m, "NelderMead", "Derivative-free simplex search, optionally box-constrained.");
    nm.def(py::init([](const NelderMeadParams& p, py::function f) {
               return std::make_unique<NelderMead>(std::make_unique<NelderMeadParams>(p), objective(std::move(f)));
           }),
           py::arg("parameters"), py::arg("objective"));
    bind_solver(nm);

    // --- Differential evolution ---
    py::enum_<DEStrategy>(m, "DEStrategy")
        .value("Rand1", DEStrategy::Rand1)
        .value("Best1", DEStrategy::Best1)
        .value("CurrentToBest1", DEStrategy::CurrentToBest1)
        .value("Rand2", DEStrategy::Rand2);
    py::class_<DifferentialEvolutionParams>(m, "DifferentialEvolutionParams")
        .def(kwargs_init<DifferentialEvolutionParams>())
        .def_readwrite("dim", &DifferentialEvolutionParams::dim)
        .def_readwrite("bounds", &DifferentialEvolutionParams::bounds)
        .def_readwrite("population_size", &DifferentialEvolutionParams::population_size)
        .def_readwrite("max_generations", &DifferentialEvolutionParams::max_generations)
        .def_readwrite("strategy", &DifferentialEvolutionParams::strategy)
        .def_readwrite("mutation_min", &DifferentialEvolutionParams::mutation_min)
        .def_readwrite("mutation_max", &DifferentialEvolutionParams::mutation_max)
        .def_readwrite("crossover", &DifferentialEvolutionParams::crossover)
        .def_readwrite("tolerance", &DifferentialEvolutionParams::tolerance)
        .def_readwrite("absolute_tolerance", &DifferentialEvolutionParams::absolute_tolerance)
        .def_readwrite("seed", &DifferentialEvolutionParams::seed)
        .def_readwrite("num_threads", &DifferentialEvolutionParams::num_threads,
                       "Threads evaluating the population. A Python objective runs under the GIL, so threads only "
                       "help if it releases the GIL (e.g. heavy numpy work).");
    py::class_<DifferentialEvolutionResult>(m, "DifferentialEvolutionResult")
        .def_readonly("x", &DifferentialEvolutionResult::x)
        .def_readonly("objective", &DifferentialEvolutionResult::objective)
        .def_readonly("generations", &DifferentialEvolutionResult::generations)
        .def_readonly("evaluations", &DifferentialEvolutionResult::evaluations)
        .def_readonly("threads_used", &DifferentialEvolutionResult::threads_used)
        .def_readonly("converged", &DifferentialEvolutionResult::converged);
    py::class_<DifferentialEvolution> de(m, "DifferentialEvolution", "Global derivative-free search in a box.");
    de.def(py::init([](const DifferentialEvolutionParams& p, py::function f) {
               return std::make_unique<DifferentialEvolution>(std::make_unique<DifferentialEvolutionParams>(p),
                                                              objective(std::move(f)));
           }),
           py::arg("parameters"), py::arg("objective"));
    bind_solver(de);
}

} // namespace nabia_bridge
