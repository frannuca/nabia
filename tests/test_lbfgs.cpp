#include <nabia/optimization/convex/LBFGS.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>

using Catch::Matchers::WithinAbs;
using Eigen::VectorXd;

namespace {

std::unique_ptr<LBFGSParams> make_params(Eigen::Index dim) {
    return std::make_unique<LBFGSParams>(LBFGSParams{dim});
}

} // namespace

TEST_CASE("lbfgs: ill-conditioned quadratic", "[optimization]") {
    // f(x) = 1/2 x'Qx - b'x with Q = diag(1, 10, 100, 1000)  ->  x = Q^-1 b
    VectorXd q(4);
    q << 1.0, 10.0, 100.0, 1000.0;
    const VectorXd b = VectorXd::Ones(4);
    LBFGS solver(
        make_params(4),
        [&](const VectorXd& x) { return 0.5 * x.dot(q.cwiseProduct(x)) - b.dot(x); },
        [&](const VectorXd& x) -> VectorXd { return q.cwiseProduct(x) - b; });
    const VectorXd x = solver.optimize(std::nullopt);  // starts from zero
    const auto& r = solver.result();
    REQUIRE(r.converged());
    CHECK((x - b.cwiseQuotient(q)).lpNorm<Eigen::Infinity>() < 1e-7);
    CHECK(x == r.x);
    CHECK(r.gradient_norm <= 1e-8);
}

TEST_CASE("lbfgs: rosenbrock with value-and-gradient callback", "[optimization]") {
    // f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum at (1, 1)
    auto rosenbrock = [](const VectorXd& z, VectorXd& grad) {
        const double x = z(0);
        const double y = z(1);
        grad(0) = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        grad(1) = 200.0 * (y - x * x);
        return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
    };
    VectorXd x0(2);
    x0 << -1.2, 1.0;
    LBFGS solver(make_params(2), rosenbrock);
    solver.optimize(x0);
    const auto& r = solver.result();
    REQUIRE(r.converged());
    CHECK_THAT(r.x(0), WithinAbs(1.0, 1e-6));
    CHECK_THAT(r.x(1), WithinAbs(1.0, 1e-6));
    CHECK(r.objective < 1e-12);
}

TEST_CASE("lbfgs: stops at max_iter", "[optimization]") {
    auto params = make_params(2);
    params->max_iter = 1;
    VectorXd x0(2);
    x0 << -1.2, 1.0;
    LBFGS solver(
        std::move(params),
        [](const VectorXd& z) { return (1 - z(0)) * (1 - z(0)) + 100 * (z(1) - z(0) * z(0)) * (z(1) - z(0) * z(0)); },
        [](const VectorXd& z) -> VectorXd {
            VectorXd g(2);
            g << -2 * (1 - z(0)) - 400 * z(0) * (z(1) - z(0) * z(0)), 200 * (z(1) - z(0) * z(0));
            return g;
        });
    solver.optimize(x0);
    CHECK(solver.result().status == LBFGSStatus::MaxIterations);
    CHECK(solver.result().iterations == 1);
}

TEST_CASE("lbfgs: invalid input", "[optimization]") {
    auto f = [](const VectorXd& x) { return x.squaredNorm(); };
    auto df = [](const VectorXd& x) -> VectorXd { return 2.0 * x; };

    LBFGS wrong_guess(make_params(3), f, df);
    CHECK_THROWS_AS(wrong_guess.optimize(VectorXd::Zero(2)), std::invalid_argument);

    LBFGS no_objective(make_params(2), LBFGS::ValueAndGradient{});
    CHECK_THROWS_AS(no_objective.optimize(std::nullopt), std::invalid_argument);

    auto bad_memory = make_params(2);
    bad_memory->memory = 0;
    LBFGS zero_memory(std::move(bad_memory), f, df);
    CHECK_THROWS_AS(zero_memory.optimize(std::nullopt), std::invalid_argument);
}

TEST_CASE("optimizers expose optimize() and result() through IOptimizer<T, R>", "[optimization]") {
    // Generic code can run any optimizer and inspect its typed result via the base class.
    auto run = []<typename T, typename R>(IOptimizer<T, R>& optimizer) -> const R& {
        optimizer.optimize(std::nullopt);
        return optimizer.result();
    };
    LBFGS solver(make_params(2), [](const VectorXd& x) { return (x - VectorXd::Ones(2)).squaredNorm(); },
                 [](const VectorXd& x) -> VectorXd { return 2.0 * (x - VectorXd::Ones(2)); });
    const LBFGSResult& r = run(solver);
    CHECK(r.converged());
    CHECK(&r == &solver.result());
    static_assert(std::is_same_v<LBFGS::Result, LBFGSResult>);
    static_assert(std::is_same_v<LBFGS::Params, LBFGSParams>);
}
