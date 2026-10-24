#include <nabia/optimization/nongradopt/DifferentialEvolution.h>
#include <nabia/optimization/nongradopt/NelderMead.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <thread>

using Catch::Matchers::WithinAbs;
using Eigen::VectorXd;

namespace {

double rosenbrock(const VectorXd& z) {
    return (1.0 - z(0)) * (1.0 - z(0)) + 100.0 * (z(1) - z(0) * z(0)) * (z(1) - z(0) * z(0));
}

// Highly multimodal; global minimum 0 at the origin, local minima near every integer point.
double rastrigin(const VectorXd& x) {
    return 10.0 * static_cast<double>(x.size()) +
           (x.array().square() - 10.0 * (2.0 * std::numbers::pi * x.array()).cos()).sum();
}

BoxConstraints cube(Eigen::Index n, double lo, double up) {
    return {std::vector<double>(static_cast<std::size_t>(n), lo), std::vector<double>(static_cast<std::size_t>(n), up)};
}

std::unique_ptr<NelderMeadParams> nm_params(Eigen::Index n, std::optional<BoxConstraints> bounds = std::nullopt) {
    auto p = std::make_unique<NelderMeadParams>();
    p->dim = n;
    p->bounds = std::move(bounds);
    return p;
}

std::unique_ptr<DifferentialEvolutionParams> de_params(Eigen::Index n, double lo, double up, int threads) {
    auto p = std::make_unique<DifferentialEvolutionParams>();
    p->dim = n;
    p->bounds = cube(n, lo, up);
    p->num_threads = threads;
    return p;
}

} // namespace

TEST_CASE("nelder-mead: unconstrained", "[optimization]") {
    SECTION("rosenbrock") {
        NelderMead nm(nm_params(2), rosenbrock);
        VectorXd x0(2);
        x0 << -1.2, 1.0;
        const VectorXd x = nm.optimize(x0);
        REQUIRE(nm.result().converged());
        CHECK_THAT(x(0), WithinAbs(1.0, 1e-5));
        CHECK_THAT(x(1), WithinAbs(1.0, 1e-5));
    }
    SECTION("shifted 6-d quadratic from the default start") {
        VectorXd c(6);
        c << 1.0, -2.0, 0.5, 3.0, -1.0, 2.0;
        NelderMead nm(nm_params(6), [&](const VectorXd& x) { return (x - c).squaredNorm(); });
        const VectorXd x = nm.optimize(std::nullopt);
        REQUIRE(nm.result().converged());
        CHECK((x - c).lpNorm<Eigen::Infinity>() < 1e-4);
    }
}

TEST_CASE("nelder-mead: box constraints", "[optimization]") {
    // Unconstrained minimum (2, -1, 0.5) lies outside [0, 1]^2 x R -> (1, 0, 0.5).
    constexpr double inf = std::numeric_limits<double>::infinity();
    VectorXd c(3);
    c << 2.0, -1.0, 0.5;
    BoxConstraints box{{0.0, 0.0, -inf}, {1.0, 1.0, inf}};
    bool outside = false;
    NelderMead nm(nm_params(3, box), [&](const VectorXd& x) {
        outside = outside || x(0) < 0.0 || x(0) > 1.0 || x(1) < 0.0 || x(1) > 1.0;
        return (x - c).squaredNorm();
    });
    const VectorXd x = nm.optimize(std::nullopt);
    REQUIRE(nm.result().converged());
    CHECK_FALSE(outside);  // f is never evaluated outside the box
    CHECK_THAT(x(0), WithinAbs(1.0, 1e-6));
    CHECK_THAT(x(1), WithinAbs(0.0, 1e-6));
    CHECK_THAT(x(2), WithinAbs(0.5, 1e-5));
}

TEST_CASE("nelder-mead: limits and invalid input", "[optimization]") {
    auto p = nm_params(2);
    p->max_evaluations = 50;
    NelderMead limited(std::move(p), rosenbrock);
    limited.optimize(std::nullopt);
    CHECK(limited.result().status == NelderMeadStatus::MaxEvaluations);
    CHECK(limited.result().evaluations <= 50 + 3);  // may finish the current iteration

    NelderMead wrong_guess(nm_params(2), rosenbrock);
    CHECK_THROWS_AS(wrong_guess.optimize(VectorXd::Zero(3)), std::invalid_argument);
    NelderMead crossed(nm_params(2, BoxConstraints{{1.0, 0.0}, {0.0, 1.0}}), rosenbrock);
    CHECK_THROWS_AS(crossed.optimize(std::nullopt), std::invalid_argument);
}

TEST_CASE("differential evolution: global minimum of rastrigin", "[optimization]") {
    // Nelder-Mead from (3.2, -2.9) gets stuck in a local minimum; DE finds the origin.
    VectorXd start(5);
    start << 3.2, -2.9, 2.1, -4.0, 1.0;
    NelderMead nm(nm_params(5, cube(5, -5.12, 5.12)), rastrigin);
    nm.optimize(start);
    CHECK(nm.result().objective > 1.0);

    auto p = de_params(5, -5.12, 5.12, 4);
    p->max_generations = 3000;
    DifferentialEvolution de(std::move(p), rastrigin);
    const VectorXd x = de.optimize(std::nullopt);
    const auto& r = de.result();
    CHECK(r.converged);
    CHECK(r.threads_used == 4);
    CHECK(r.objective < 1e-6);
    CHECK(x.lpNorm<Eigen::Infinity>() < 1e-3);
    CHECK(r.evaluations == 75 * (r.generations + 1));  // NP = 15 * dim per generation
}

TEST_CASE("differential evolution: every strategy solves rosenbrock", "[optimization]") {
    for (DEStrategy s : {DEStrategy::Rand1, DEStrategy::Best1, DEStrategy::CurrentToBest1, DEStrategy::Rand2}) {
        auto p = de_params(2, -5.0, 5.0, 2);
        p->strategy = s;
        p->max_generations = 5000;
        p->tolerance = 0.0;
        p->absolute_tolerance = 1e-12;
        DifferentialEvolution de(std::move(p), rosenbrock);
        const VectorXd x = de.optimize(std::nullopt);
        CHECK(de.result().converged);
        CHECK_THAT(x(0), WithinAbs(1.0, 1e-4));
        CHECK_THAT(x(1), WithinAbs(1.0, 1e-4));
    }
}

TEST_CASE("differential evolution: results do not depend on the number of threads", "[optimization]") {
    auto run = [](int threads) {
        auto p = de_params(4, -5.12, 5.12, threads);
        p->max_generations = 200;
        DifferentialEvolution de(std::move(p), rastrigin);
        const VectorXd x = de.optimize(std::nullopt);
        return std::pair{x, de.result().objective};
    };
    const auto [x1, f1] = run(1);
    const auto [x3, f3] = run(3);
    const auto [x8, f8] = run(8);
    CHECK(x1 == x3);
    CHECK(x1 == x8);
    CHECK(f1 == f8);
}

TEST_CASE("differential evolution: population is evaluated concurrently", "[optimization]") {
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    auto slow = [&](const VectorXd& x) {
        const int now = ++active;
        int seen = max_active.load();
        while (now > seen && !max_active.compare_exchange_weak(seen, now)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        --active;
        return x.squaredNorm();
    };
    auto p = de_params(2, -1.0, 1.0, 4);
    p->population_size = 16;
    p->max_generations = 3;
    DifferentialEvolution de(std::move(p), slow);
    de.optimize(std::nullopt);
    CHECK(de.result().threads_used == 4);
    CHECK(max_active.load() > 1);
    CHECK(max_active.load() <= 4);
}

TEST_CASE("differential evolution: initial guess, errors and invalid input", "[optimization]") {
    SECTION("an optimal initial guess is kept") {
        auto p = de_params(3, -1.0, 1.0, 2);
        p->max_generations = 1;
        DifferentialEvolution de(std::move(p), [](const VectorXd& x) { return (x.array() - 0.25).square().sum(); });
        const VectorXd x = de.optimize(VectorXd::Constant(3, 0.25));
        CHECK(de.result().objective == 0.0);
        CHECK(x == VectorXd::Constant(3, 0.25));
    }
    SECTION("exceptions from worker threads reach the caller") {
        DifferentialEvolution de(de_params(2, -1.0, 1.0, 4), [](const VectorXd& x) -> double {
            if (x(0) > 0.5) {
                throw std::runtime_error("objective failed");
            }
            return x.squaredNorm();
        });
        CHECK_THROWS_AS(de.optimize(std::nullopt), std::runtime_error);
    }
    SECTION("invalid parameters") {
        auto unbounded = std::make_unique<DifferentialEvolutionParams>();
        unbounded->dim = 2;
        DifferentialEvolution no_bounds(std::move(unbounded), rastrigin);
        CHECK_THROWS_AS(no_bounds.optimize(std::nullopt), std::invalid_argument);

        auto tiny = de_params(2, -1.0, 1.0, 1);
        tiny->strategy = DEStrategy::Rand2;
        tiny->population_size = 5;
        DifferentialEvolution small_population(std::move(tiny), rastrigin);
        CHECK_THROWS_AS(small_population.optimize(std::nullopt), std::invalid_argument);

        auto bad_cr = de_params(2, -1.0, 1.0, 1);
        bad_cr->crossover = 1.5;
        DifferentialEvolution bad_crossover(std::move(bad_cr), rastrigin);
        CHECK_THROWS_AS(bad_crossover.optimize(std::nullopt), std::invalid_argument);
    }
}
