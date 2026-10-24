#pragma once

#include <cstddef>
#include <span>

namespace nabia::stats {

/// Arithmetic mean. Throws std::invalid_argument if `x` is empty.
[[nodiscard]] double mean(std::span<const double> x);

/// Variance. `ddof` = delta degrees of freedom (0 = population, 1 = sample).
/// Throws std::invalid_argument if x.size() <= ddof.
[[nodiscard]] double variance(std::span<const double> x, std::size_t ddof = 1);

/// Standard deviation, sqrt(variance(x, ddof)).
[[nodiscard]] double stddev(std::span<const double> x, std::size_t ddof = 1);

/// Sample autocorrelation at the given lag (biased estimator, as used in ACF plots).
/// Throws std::invalid_argument if lag >= x.size() or the series is constant.
[[nodiscard]] double autocorrelation(std::span<const double> x, std::size_t lag);

} // namespace nabia::stats
