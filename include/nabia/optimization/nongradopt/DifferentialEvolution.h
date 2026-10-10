//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_DIFFERENTIALEVOLUTION_H
#define NABIA_DIFFERENTIALEVOLUTION_H

#include "nabia/optimization/IOptimizer.h"
#include "nabia/optimization/convex/Constraints.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <Eigen/Dense>

/// Mutation strategies; all use binomial crossover.
enum class DEStrategy {
    Rand1,          // v = x_r1 + F (x_r2 - x_r3)                      robust, explores well
    Best1,          // v = x_best + F (x_r1 - x_r2)                    converges fast, may get trapped
    CurrentToBest1, // v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)    a compromise
    Rand2,          // v = x_r1 + F (x_r2 - x_r3) + F (x_r4 - x_r5)    most exploratory
};

struct DifferentialEvolutionParams {
    Eigen::Index dim;
    BoxConstraints bounds;           // required and finite: the search region and the sampling box
    int population_size = 0;         // NP; 0 = 15 * dim. At least 4 (6 for Rand2)
    int max_generations = 1000;
    DEStrategy strategy = DEStrategy::Rand1;
    double mutation_min = 0.5;       // F is drawn uniformly in [mutation_min, mutation_max] each
    double mutation_max = 1.0;       // generation (dithering); set them equal for a constant F
    double crossover = 0.9;          // CR in [0, 1]
    double tolerance = 1e-8;         // stop when std(f) <= absolute_tolerance + tolerance * |mean(f)|
    double absolute_tolerance = 0.0;
    std::uint64_t seed = 42;         // runs are reproducible for a given seed, whatever num_threads is
    /// Level of parallelism for evaluating the population: number of threads,
    /// including the calling thread. 0 = std::thread::hardware_concurrency(),
    /// 1 = serial. The objective must be safe to call concurrently.
    int num_threads = 0;
};

struct DifferentialEvolutionResult {
    Eigen::VectorXd x;
    double objective = 0.0;
    int generations = 0;
    int evaluations = 0;
    int threads_used = 1;
    bool converged = false;   // population spread below tolerance (otherwise max_generations was hit)
};

/// Differential Evolution (Storn & Price 1997) for global minimization of
///   min f(x)  s.t.  l <= x <= u.
///
/// Derivative free and population based: suited to multimodal, noisy or
/// non-smooth objectives. Each generation builds one trial vector per member
/// by mutation and binomial crossover, and keeps the trial if it is no worse
/// (greedy selection). Trial components that leave the box are moved halfway
/// between the parent and the violated bound.
///
/// Trial vectors are generated sequentially from a seeded RNG, and then the
/// whole population is evaluated in parallel on `num_threads` threads. Results
/// therefore do not depend on the number of threads.
class DifferentialEvolution: public IOptimizer<DifferentialEvolutionParams, DifferentialEvolutionResult> {
public:
    using Objective = std::function<double(const Eigen::VectorXd&)>;

    DifferentialEvolution(std::unique_ptr<DifferentialEvolutionParams> parameters, Objective objective);

    /// If `initial_guess` is given (projected onto the box), it replaces one
    /// member of the random initial population. Returns the best member;
    /// details in result(). Throws std::invalid_argument on bad parameters and
    /// rethrows any exception thrown by the objective.
    Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) override;

    [[nodiscard]] const DifferentialEvolutionResult& result() const override { return result_; }

private:
    Objective objective_;
    DifferentialEvolutionResult result_;
};

#endif // NABIA_DIFFERENTIALEVOLUTION_H
