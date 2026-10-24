//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_IOPTIMIZER_H
#define NABIA_IOPTIMIZER_H

#include <memory>
#include <optional>
#include <utility>
#include <Eigen/Dense>

/// Base of every nabia optimizer.
///   T: parameter type, owned through a std::unique_ptr;
///   R: result type, reported by result() after optimize().
template<typename T, typename R> class IOptimizer {
public:
    using Params = T;
    using Result = R;

    explicit IOptimizer(std::unique_ptr<T> parameters) : parameters(std::move(parameters)) {}
    virtual ~IOptimizer() = default;

    IOptimizer(const IOptimizer&) = delete;
    IOptimizer& operator=(const IOptimizer&) = delete;
    IOptimizer(IOptimizer&&) noexcept = default;
    IOptimizer& operator=(IOptimizer&&) noexcept = default;

    /// Runs the optimizer and returns the solution. Without an initial guess,
    /// the solver starts from its default (zero for the solvers in nabia).
    virtual Eigen::VectorXd optimize(std::optional<Eigen::VectorXd> initial_guess) = 0;

    /// Details of the last optimize() call: solution, objective, convergence
    /// flags and solver-specific diagnostics. Default-constructed before the first call.
    [[nodiscard]] virtual const R& result() const = 0;

protected:
    std::unique_ptr<T> parameters;
};

#endif // NABIA_IOPTIMIZER_H
