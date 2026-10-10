#include <nabia/optimization/convex/AugmentedLagrangian.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <memory>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Eigen::VectorXd;

namespace {

constexpr double kTol = 1e-5;

std::unique_ptr<ConvexOptimizationParams> make_params(Eigen::Index dim, std::optional<Constraints> c = std::nullopt) {
    return std::make_unique<ConvexOptimizationParams>(ConvexOptimizationParams{dim, 1e-8, 100, std::move(c)});
}

// f(x) = |x - c|^2
AugmentedLagrangian distance_to(const VectorXd& c, std::optional<Constraints> constraints) {
    return {make_params(c.size(), std::move(constraints)),
            [c](const VectorXd& x) { return (x - c).squaredNorm(); },
            [c](const VectorXd& x) -> VectorXd { return 2.0 * (x - c); }};
}

} // namespace

TEST_CASE("augmented lagrangian: unconstrained quadratic", "[optimization]") {
    VectorXd c(3);
    c << 1.0, -2.0, 3.0;
    auto solver = distance_to(c, std::nullopt);
    const VectorXd x = solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK((x - c).lpNorm<Eigen::Infinity>() < kTol);
    CHECK(x == r.x);
    CHECK(r.lambda_eq.size() == 0);
    CHECK(r.mu_ineq.size() == 0);
}

TEST_CASE("augmented lagrangian: equality constraint", "[optimization]") {
    // min x^2 + y^2  s.t.  x + y = 1  ->  x = y = 1/2, lambda = -1
    LinearConstraint lc;
    lc.A_eq.resize(1, 2);
    lc.A_eq << 1.0, 1.0;
    lc.b_eq.resize(1);
    lc.b_eq << 1.0;
    auto solver = distance_to(VectorXd::Zero(2), lc);
    solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(r.x(0), WithinAbs(0.5, kTol));
    CHECK_THAT(r.x(1), WithinAbs(0.5, kTol));
    CHECK_THAT(r.lambda_eq(0), WithinAbs(-1.0, kTol));
}

TEST_CASE("augmented lagrangian: active and inactive inequality", "[optimization]") {
    // min (x-2)^2 + (y-2)^2  s.t.  x + y <= b
    VectorXd c(2);
    c << 2.0, 2.0;
    LinearConstraint lc;
    lc.A_ineq.resize(1, 2);
    lc.A_ineq << 1.0, 1.0;
    lc.b_ineq.resize(1);

    SECTION("active: b = 2 -> x = y = 1, mu = 2") {
        lc.b_ineq << 2.0;
        auto solver = distance_to(c, lc);
        solver.optimize(std::nullopt);
        const auto& r = solver.result();
        REQUIRE(r.converged);
        CHECK_THAT(r.x(0), WithinAbs(1.0, kTol));
        CHECK_THAT(r.x(1), WithinAbs(1.0, kTol));
        CHECK_THAT(r.mu_ineq(0), WithinAbs(2.0, kTol));
    }
    SECTION("inactive: b = 10 -> x = y = 2, mu = 0") {
        lc.b_ineq << 10.0;
        auto solver = distance_to(c, lc);
        solver.optimize(std::nullopt);
        const auto& r = solver.result();
        REQUIRE(r.converged);
        CHECK_THAT(r.x(0), WithinAbs(2.0, kTol));
        CHECK_THAT(r.x(1), WithinAbs(2.0, kTol));
        CHECK_THAT(r.mu_ineq(0), WithinAbs(0.0, kTol));
    }
}

TEST_CASE("augmented lagrangian: projection onto the simplex", "[optimization]") {
    // min |x - c|^2  s.t.  sum x = 1, x >= 0, c = (1, -1, 0.5)  ->  x = (0.75, 0, 0.25)
    VectorXd c(3);
    c << 1.0, -1.0, 0.5;
    LinearConstraint lc;
    lc.A_eq = Eigen::MatrixXd::Ones(1, 3);
    lc.b_eq = VectorXd::Ones(1);
    lc.A_ineq = -Eigen::MatrixXd::Identity(3, 3);
    lc.b_ineq = VectorXd::Zero(3);
    auto solver = distance_to(c, lc);
    solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(r.x(0), WithinAbs(0.75, kTol));
    CHECK_THAT(r.x(1), WithinAbs(0.0, kTol));
    CHECK_THAT(r.x(2), WithinAbs(0.25, kTol));
    CHECK(lc.is_satisfied(r.x, kTol));
    // KKT: 2(x - c) + lambda * 1 - mu = 0, mu_i x_i = 0
    CHECK_THAT(r.lambda_eq(0), WithinAbs(0.5, kTol));
    CHECK_THAT(r.mu_ineq(1), WithinAbs(2.5, kTol));
}

TEST_CASE("augmented lagrangian: box constraints", "[optimization]") {
    // min (x-2)^2 + (y+1)^2 + (z-0.5)^2 on [0,1] x [0,1] x (-inf, inf)  ->  (1, 0, 0.5)
    constexpr double inf = std::numeric_limits<double>::infinity();
    VectorXd c(3);
    c << 2.0, -1.0, 0.5;
    BoxConstraints box{{0.0, 0.0, -inf}, {1.0, 1.0, inf}};
    auto solver = distance_to(c, box);
    solver.optimize(std::nullopt);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(r.x(0), WithinAbs(1.0, kTol));
    CHECK_THAT(r.x(1), WithinAbs(0.0, kTol));
    CHECK_THAT(r.x(2), WithinAbs(0.5, kTol));
    // rows: x0 <= 1, x1 <= 1, -x0 <= 0, -x1 <= 0
    REQUIRE(r.mu_ineq.size() == 4);
    CHECK_THAT(r.mu_ineq(0), WithinAbs(2.0, kTol));
    CHECK_THAT(r.mu_ineq(3), WithinAbs(2.0, kTol));
}

TEST_CASE("augmented lagrangian: non-quadratic objective", "[optimization]") {
    // min exp(x) + exp(y)  s.t.  x + y = 0  ->  x = y = 0, lambda = -1
    LinearConstraint lc;
    lc.A_eq = Eigen::MatrixXd::Ones(1, 2);
    lc.b_eq = VectorXd::Zero(1);
    VectorXd x0(2);
    x0 << 1.0, -3.0;
    AugmentedLagrangian solver(
        make_params(2, lc), [](const VectorXd& x) { return x.array().exp().sum(); },
        [](const VectorXd& x) -> VectorXd { return x.array().exp(); });
    solver.optimize(x0);
    const auto& r = solver.result();
    REQUIRE(r.converged);
    CHECK_THAT(r.x(0), WithinAbs(0.0, kTol));
    CHECK_THAT(r.x(1), WithinAbs(0.0, kTol));
    CHECK_THAT(r.lambda_eq(0), WithinAbs(-1.0, kTol));
}

TEST_CASE("augmented lagrangian: invalid input", "[optimization]") {
    LinearConstraint bad;
    bad.A_eq = Eigen::MatrixXd::Ones(1, 3);  // 3 columns, dim is 2
    bad.b_eq = VectorXd::Ones(1);
    auto wrong_constraint = distance_to(VectorXd::Zero(2), bad);
    CHECK_THROWS_AS(wrong_constraint.optimize(std::nullopt), std::invalid_argument);

    auto crossed_box = distance_to(VectorXd::Zero(2), BoxConstraints{{1.0, 0.0}, {0.0, 1.0}});
    CHECK_THROWS_AS(crossed_box.optimize(std::nullopt), std::invalid_argument);

    AugmentedLagrangian wrong_guess(
        make_params(3), [](const VectorXd& x) { return x.squaredNorm(); },
        [](const VectorXd& x) -> VectorXd { return 2.0 * x; });
    CHECK_THROWS_AS(wrong_guess.optimize(VectorXd::Zero(2)), std::invalid_argument);
}
