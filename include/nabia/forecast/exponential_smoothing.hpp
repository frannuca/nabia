#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nabia::forecast {

/// Simple Exponential Smoothing (SES): a level-only forecasting model.
///
///   level[t] = alpha * y[t] + (1 - alpha) * level[t-1]
///   forecast(h) = level[T]   for every horizon h
///
/// Suitable for series without trend or seasonality.
class SimpleExponentialSmoothing {
public:
    /// Throws std::invalid_argument if alpha is not in (0, 1].
    explicit SimpleExponentialSmoothing(double alpha);

    /// Fit the model on a series. Throws std::invalid_argument if `y` is empty.
    void fit(std::span<const double> y);

    /// Forecast `horizon` steps ahead. Throws std::logic_error if not fitted.
    [[nodiscard]] std::vector<double> forecast(std::size_t horizon) const;

    /// One-step-ahead in-sample fitted values (fitted[t] is the forecast of y[t]).
    [[nodiscard]] const std::vector<double>& fitted() const noexcept { return fitted_; }

    /// Sum of squared one-step-ahead errors on the training data.
    [[nodiscard]] double sse() const noexcept { return sse_; }

    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] bool is_fitted() const noexcept { return is_fitted_; }

private:
    double alpha_;
    double level_ = 0.0;
    double sse_ = 0.0;
    bool is_fitted_ = false;
    std::vector<double> fitted_;
};

} // namespace nabia::forecast
