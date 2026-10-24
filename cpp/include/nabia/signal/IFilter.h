//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_IFILTER_H
#define NABIA_IFILTER_H
#include <memory>
#include <utility>
#include <Eigen/Dense>

/// Common interface of the nabia signal filters, parameterized by the
/// filter's parameter struct T.
///
/// Filters are causal: output sample t depends only on inputs 0..t. Two ways to use them:
///   - streaming: process() one sample at a time; the filter keeps its state
///     between calls until reset();
///   - batch: apply() filters a whole signal in place, starting from a fresh
///     state, without touching the streaming state (hence const).
template<typename T>
class IFilter {
public:
    explicit IFilter(std::unique_ptr<T> parameters) : parameters_(std::move(parameters)) {}
    virtual ~IFilter() = default;

    IFilter(const IFilter&) = delete;
    IFilter& operator=(const IFilter&) = delete;
    IFilter(IFilter&&) noexcept = default;
    IFilter& operator=(IFilter&&) noexcept = default;

    /// Filters one sample, advancing the internal state.
    virtual double process(double sample) = 0;

    /// Clears the internal state, as if no sample had been processed.
    virtual void reset() = 0;

    /// Filters `signal` in place from a fresh state.
    virtual void apply(Eigen::VectorXd& signal) const = 0;

    [[nodiscard]] const T& parameters() const { return *parameters_; }

protected:
    std::unique_ptr<T> parameters_;
};

#endif // NABIA_IFILTER_H
