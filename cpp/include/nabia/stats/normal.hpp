#pragma once

namespace nabia::stats {

/// Standard normal density phi(x).
[[nodiscard]] double normal_pdf(double x);

/// Standard normal cumulative distribution Phi(x).
[[nodiscard]] double normal_cdf(double x);

/// Inverse of Phi: the z with Phi(z) = p. Accurate to about 1e-15.
/// Throws std::invalid_argument unless 0 < p < 1.
[[nodiscard]] double normal_quantile(double p);

} // namespace nabia::stats
