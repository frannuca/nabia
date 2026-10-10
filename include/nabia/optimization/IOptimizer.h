//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_IOPTIMIZER_H
#define NABIA_IOPTIMIZER_H

#include <memory>
#include <optional>
#include <utility>
#include <Eigen/Dense>

template<typename T> class IOptimizer {
public:
    explicit IOptimizer(std::unique_ptr<T> parameters) : parameters(std::move(parameters)) {}
    virtual ~IOptimizer() = default;

    IOptimizer(const IOptimizer&) = delete;
    IOptimizer& operator=(const IOptimizer&) = delete;
    IOptimizer(IOptimizer&&) noexcept = default;
    IOptimizer& operator=(IOptimizer&&) noexcept = default;

    /// Runs the optimizer and returns the solution. Without an initial guess,
    /// the solver starts from its default (zero for the solvers in nabia).
    virtual Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) = 0;

protected:
    std::unique_ptr<T> parameters;
};

#endif // NABIA_IOPTIMIZER_H
