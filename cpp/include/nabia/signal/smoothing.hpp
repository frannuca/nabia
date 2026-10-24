#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nabia::signal {

/// Trailing simple moving average. Output has x.size() - window + 1 elements.
/// Throws std::invalid_argument if window == 0 or window > x.size().
[[nodiscard]] std::vector<double> moving_average(std::span<const double> x, std::size_t window);

/// Exponential moving average: y[0] = x[0], y[t] = alpha*x[t] + (1-alpha)*y[t-1].
/// Throws std::invalid_argument if alpha is not in (0, 1].
[[nodiscard]] std::vector<double> exponential_moving_average(std::span<const double> x,
                                                             double alpha);

/// Discrete difference of the given order: y[t] = x[t+1] - x[t] (applied `order` times).
[[nodiscard]] std::vector<double> diff(std::span<const double> x, std::size_t order = 1);

} // namespace nabia::signal
