//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/optimization/nongradopt/DifferentialEvolution.h"

#include "box.hpp"
#include "parallel_evaluator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

/// Picks `k` distinct indices in [0, np) that also differ from `exclude`.
template <std::size_t K>
std::array<Index, K> pick_distinct(std::mt19937_64& rng, Index np, Index exclude) {
    std::uniform_int_distribution<Index> dist(0, np - 1);
    std::array<Index, K> picked{};
    for (std::size_t k = 0; k < K; ++k) {
        Index r = 0;
        bool fresh = false;
        while (!fresh) {
            r = dist(rng);
            fresh = r != exclude && std::find(picked.begin(), picked.begin() + static_cast<std::ptrdiff_t>(k), r) ==
                                        picked.begin() + static_cast<std::ptrdiff_t>(k);
        }
        picked[k] = r;
    }
    return picked;
}

} // namespace

DifferentialEvolution::DifferentialEvolution(std::unique_ptr<DifferentialEvolutionParams> parameters,
                                             Objective objective)
    : IOptimizer(std::move(parameters)), objective_(std::move(objective)) {}

Eigen::VectorXd DifferentialEvolution::optimize(std::optional<Eigen::VectorXd> initial_guess) {
    if (!parameters) {
        throw std::invalid_argument("DifferentialEvolution: parameters are null");
    }
    if (!objective_) {
        throw std::invalid_argument("DifferentialEvolution: objective is required");
    }
    const DifferentialEvolutionParams& p = *parameters;
    const Index n = p.dim;
    if (n <= 0 || (initial_guess && initial_guess->size() != n)) {
        throw std::invalid_argument(
            "DifferentialEvolution: `dim` must be > 0 and the initial guess must have `dim` entries");
    }
    const nabia::nongradopt::detail::Box box = nabia::nongradopt::detail::make_box(p.bounds, n, "DifferentialEvolution");
    if (!box.finite()) {
        throw std::invalid_argument("DifferentialEvolution: lower and upper bounds are required and must be finite");
    }
    // Member i plus the distinct random members the strategy draws.
    const Index min_population = p.strategy == DEStrategy::Rand2 ? 6 : 4;
    const Index np = p.population_size > 0 ? Index{p.population_size} : std::max(min_population, 15 * n);
    if (np < min_population) {
        throw std::invalid_argument("DifferentialEvolution: population too small for the chosen strategy");
    }
    if (!(p.mutation_min > 0.0 && p.mutation_max >= p.mutation_min && p.mutation_max <= 2.0) ||
        !(p.crossover >= 0.0 && p.crossover <= 1.0) || !(p.tolerance >= 0.0 && p.absolute_tolerance >= 0.0) ||
        p.max_generations < 0 || p.num_threads < 0) {
        throw std::invalid_argument("DifferentialEvolution: invalid mutation, crossover, tolerance or thread settings");
    }

    const int hardware = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int threads = std::min(p.num_threads == 0 ? hardware : p.num_threads, static_cast<int>(np));
    nabia::nongradopt::detail::ParallelEvaluator evaluator(threads, objective_);

    result_ = DifferentialEvolutionResult{};
    result_.threads_used = evaluator.num_threads();
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Initial population: uniform in the box, one member per column.
    const VectorXd width = box.upper - box.lower;
    MatrixXd population(n, np);
    for (Index j = 0; j < np; ++j) {
        for (Index i = 0; i < n; ++i) {
            population(i, j) = box.lower(i) + unit(rng) * width(i);
        }
    }
    if (initial_guess) {
        population.col(0) = box.project(*initial_guess);
    }
    VectorXd fitness(np);
    evaluator.evaluate(population, fitness);
    result_.evaluations = static_cast<int>(np);

    MatrixXd trials(n, np);
    VectorXd trial_fitness(np);
    std::uniform_int_distribution<Index> any_coordinate(0, n - 1);

    int generation = 0;
    for (; generation < p.max_generations; ++generation) {
        const double mean = fitness.mean();
        const double stddev = std::sqrt((fitness.array() - mean).square().mean());
        if (std::isfinite(stddev) && stddev <= p.absolute_tolerance + p.tolerance * std::abs(mean)) {
            result_.converged = true;
            break;
        }

        Index best = 0;
        fitness.minCoeff(&best);
        const double F = p.mutation_min + (p.mutation_max - p.mutation_min) * unit(rng);

        // Build all trial vectors first (sequential RNG => thread-count independent).
        for (Index j = 0; j < np; ++j) {
            VectorXd mutant(n);
            switch (p.strategy) {
            case DEStrategy::Rand1: {
                const auto r = pick_distinct<3>(rng, np, j);
                mutant = population.col(r[0]) + F * (population.col(r[1]) - population.col(r[2]));
                break;
            }
            case DEStrategy::Best1: {
                const auto r = pick_distinct<2>(rng, np, j);
                mutant = population.col(best) + F * (population.col(r[0]) - population.col(r[1]));
                break;
            }
            case DEStrategy::CurrentToBest1: {
                const auto r = pick_distinct<2>(rng, np, j);
                mutant = population.col(j) + F * (population.col(best) - population.col(j)) +
                         F * (population.col(r[0]) - population.col(r[1]));
                break;
            }
            case DEStrategy::Rand2: {
                const auto r = pick_distinct<5>(rng, np, j);
                mutant = population.col(r[0]) + F * (population.col(r[1]) - population.col(r[2])) +
                         F * (population.col(r[3]) - population.col(r[4]));
                break;
            }
            }

            // Binomial crossover; coordinate j_rand always comes from the mutant.
            const Index j_rand = any_coordinate(rng);
            for (Index i = 0; i < n; ++i) {
                double v = (i == j_rand || unit(rng) < p.crossover) ? mutant(i) : population(i, j);
                // Bounce back halfway between the parent and the violated bound.
                if (v < box.lower(i)) {
                    v = 0.5 * (box.lower(i) + population(i, j));
                } else if (v > box.upper(i)) {
                    v = 0.5 * (box.upper(i) + population(i, j));
                }
                trials(i, j) = v;
            }
        }

        evaluator.evaluate(trials, trial_fitness);  // parallel
        result_.evaluations += static_cast<int>(np);

        // Greedy selection; ties go to the trial to keep the population moving on plateaus.
        for (Index j = 0; j < np; ++j) {
            if (trial_fitness(j) <= fitness(j)) {
                population.col(j) = trials.col(j);
                fitness(j) = trial_fitness(j);
            }
        }
    }

    Index best = 0;
    result_.objective = fitness.minCoeff(&best);
    result_.x = population.col(best);
    result_.generations = generation;
    return result_.x;
}
