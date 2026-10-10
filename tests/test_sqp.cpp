#include <nabia/optimization/convex/AugmentedLagrangian.h>
#include <nabia/optimization/convex/SQP.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <memory>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

constexpr double kTol = 1e-6;

std::unique_ptr<ConvexOptimizationParams> make_params(Eigen::Index dim, std::optional<Constraints> c = std::nullopt) {
    return std::make_unique<ConvexOptimizationParams>(ConvexOptimizationParams{dim, 1e-8, 100, std::move(c)});
}

// f(x) = |x - c|^2
SQP distance_to(const VectorXd& c, std::optional<Constraints> constraints) {
    return {make_params(c.size(), std::move(constraints)),
            [c](const VectorXd& x) { return (x - c).squaredNorm(); },
            [c](const VectorXd& x) -> VectorXd { return 2.0 * (x - c); }};
}

double rosenbrock(const VectorXd& z) {
    return (1.0 - z(0)) * (1.0 - z(0)) + 100.0 * (z(1) - z(0) * z(0)) * (z(1) - z(0) * z(0));
}

VectorXd rosenbrock_gradient(const VectorXd& z) {
    VectorXd g(2);
    g << -2.0 * (1.0 - z(0)) - 400.0 * z(0) * (z(1) - z(0) * z(0)), 200.0 * (z(1) - z(0) * z(0));
    return g;
}

} // namespace

TEST_CASE("sqp: unconstrained problems", "[optimization]") {
    SECTION("quadratic") {
        VectorXd c(3);
        c << 1.0, -2.0, 3.0;
        auto solver = distance_to(c, std::nullopt);
        const VectorXd x = solver.optimize(std::nullopt);
        REQUIRE(solver.result().converged);
        CHECK((x - c).lpNorm<Eigen::Infinity>() < kTol);
    }
    SECTION("rosenbrock") {
        SQP solver(make_params(2), rosenbrock, rosenbrock_gradient);
        VectorXd x0(2);
        x0 << -1.2, 1.0;
        const VectorXd x = solver.optimize(x0);
        REQUIRE(solver.result().converged);
        CHECK_THAT(x(0), WithinAbs(1.0, kTol));
        CHECK_THAT(x(1), WithinAbs(1.0, kTol));
    }
}

TEST_CASE("sqp: equality constraint", "[optimization]") {
    // min x^2 + y^2  s.t.  x + y = 1  ->  x = y = 1/2, lambda = -1
    LinearConstraint lc;
    lc.A_eq = MatrixXd::Ones(1, 2);
    lc.b_eq = VectorXd::Ones(1);
    auto solver = distance_to(VectorXd::Zero(2), lc);
    const VectorXd x = solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(x(0), WithinAbs(0.5, kTol));
    CHECK_THAT(x(1), WithinAbs(0.5, kTol));
    CHECK_THAT(r.lambda_eq(0), WithinAbs(-1.0, kTol));
}

TEST_CASE("sqp: projection onto the simplex", "[optimization]") {
    // min |x - c|^2  s.t.  sum x = 1, x >= 0, c = (1, -1, 0.5)  ->  x = (0.75, 0, 0.25)
    VectorXd c(3);
    c << 1.0, -1.0, 0.5;
    LinearConstraint lc;
    lc.A_eq = MatrixXd::Ones(1, 3);
    lc.b_eq = VectorXd::Ones(1);
    lc.A_ineq = -MatrixXd::Identity(3, 3);
    lc.b_ineq = VectorXd::Zero(3);
    auto solver = distance_to(c, lc);
    const VectorXd x = solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(x(0), WithinAbs(0.75, kTol));
    CHECK_THAT(x(1), WithinAbs(0.0, kTol));
    CHECK_THAT(x(2), WithinAbs(0.25, kTol));
    CHECK_THAT(r.lambda_eq(0), WithinAbs(0.5, kTol));
    CHECK_THAT(r.mu_ineq(1), WithinAbs(2.5, kTol));
}

TEST_CASE("sqp: box constraints", "[optimization]") {
    constexpr double inf = std::numeric_limits<double>::infinity();
    VectorXd c(3);
    c << 2.0, -1.0, 0.5;
    auto solver = distance_to(c, BoxConstraints{{0.0, 0.0, -inf}, {1.0, 1.0, inf}});
    const VectorXd x = solver.optimize(std::nullopt);
    REQUIRE(solver.result().converged);
    CHECK_THAT(x(0), WithinAbs(1.0, kTol));
    CHECK_THAT(x(1), WithinAbs(0.0, kTol));
    CHECK_THAT(x(2), WithinAbs(0.5, kTol));
}

TEST_CASE("sqp: constrained rosenbrock from an infeasible start", "[optimization]") {
    // min rosenbrock  s.t.  x + y <= 1. The constraint is active; along x + y = 1
    // the stationarity condition gives a unique minimizer.
    LinearConstraint lc;
    lc.A_ineq = MatrixXd::Ones(1, 2);
    lc.b_ineq = VectorXd::Ones(1);
    SQP solver(make_params(2, lc), rosenbrock, rosenbrock_gradient);
    VectorXd x0(2);
    x0 << 2.0, 2.0;  // violates x + y <= 1
    const VectorXd x = solver.optimize(x0);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(x.sum(), WithinAbs(1.0, kTol));
    // KKT: grad f + mu (1, 1) = 0 with mu > 0
    const VectorXd g = rosenbrock_gradient(x);
    CHECK_THAT(g(0), WithinAbs(g(1), 1e-5));
    CHECK(r.mu_ineq(0) > 0.0);
    CHECK_THAT(g(0) + r.mu_ineq(0), WithinAbs(0.0, 1e-5));
}

TEST_CASE("sqp: needs fewer objective evaluations than the augmented lagrangian", "[optimization]") {
    // exp-sum objective on the simplex: smooth, non-quadratic, with active bounds.
    VectorXd w(5);
    w << 1.0, 2.0, 0.5, 3.0, 1.5;
    LinearConstraint lc;
    lc.A_eq = MatrixXd::Ones(1, 5);
    lc.b_eq = VectorXd::Ones(1);
    lc.A_ineq = -MatrixXd::Identity(5, 5);
    lc.b_ineq = VectorXd::Zero(5);

    int sqp_calls = 0;
    int al_calls = 0;
    auto f = [&w](int& calls) {
        return [&w, &calls](const VectorXd& x) {
            ++calls;
            return (w.array() * x.array()).exp().sum() - 3.0 * x(3);
        };
    };
    auto df = [&w](const VectorXd& x) -> VectorXd {
        VectorXd g = w.array() * (w.array() * x.array()).exp();
        g(3) -= 3.0;
        return g;
    };
    SQP sqp(make_params(5, lc), f(sqp_calls), df);
    AugmentedLagrangian al(make_params(5, lc), f(al_calls), df);
    const VectorXd x_sqp = sqp.optimize(std::nullopt);
    const VectorXd x_al = al.optimize(std::nullopt);
    REQUIRE(sqp.result().converged);
    REQUIRE(al.result().converged);
    CHECK((x_sqp - x_al).lpNorm<Eigen::Infinity>() < kTol);
    CHECK(sqp_calls * 5 < al_calls);
}

TEST_CASE("sqp: invalid input", "[optimization]") {
    LinearConstraint bad;
    bad.A_eq = MatrixXd::Ones(1, 3);  // 3 columns, dim is 2
    bad.b_eq = VectorXd::Ones(1);
    auto wrong_constraint = distance_to(VectorXd::Zero(2), bad);
    CHECK_THROWS_AS(wrong_constraint.optimize(std::nullopt), std::invalid_argument);

    auto crossed_box = distance_to(VectorXd::Zero(2), BoxConstraints{{1.0, 0.0}, {0.0, 1.0}});
    CHECK_THROWS_AS(crossed_box.optimize(std::nullopt), std::invalid_argument);

    auto wrong_guess = distance_to(VectorXd::Zero(3), std::nullopt);
    CHECK_THROWS_AS(wrong_guess.optimize(VectorXd::Zero(2)), std::invalid_argument);
}
