#pragma once

#include "nabia/forecast/IForecaster.h"

#include <memory>
#include <Eigen/Dense>

namespace nabia::forecast {

struct SimpleExponentialSmoothingParams {
    double alpha = 0.3;  // smoothing factor in (0, 1]
};

/// Simple Exponential Smoothing (SES): a level-only forecasting model.
///
///   level[t] = alpha * y[t] + (1 - alpha) * level[t-1]
///   forecast(h) = level[T]   for every horizon h
///
/// Suitable for series without trend or seasonality. Forecast variances follow
/// the equivalent ETS(A,N,N) model: sigma^2 (1 + (h - 1) alpha^2), with sigma^2
/// estimated by the mean squared one-step error.
class SimpleExponentialSmoothing: public IForecaster<SimpleExponentialSmoothingParams> {
public:
    /// Throws std::invalid_argument if alpha is not in (0, 1].
    explicit SimpleExponentialSmoothing(std::unique_ptr<SimpleExponentialSmoothingParams> parameters);
    /// Convenience: SimpleExponentialSmoothing(0.5).
    explicit SimpleExponentialSmoothing(double alpha);

    /// Fit the model on a series. Throws std::invalid_argument if `y` is empty or not finite.
    void fit(const Eigen::VectorXd& y) override;

    /// Forecast `horizon` steps ahead. Throws std::logic_error if not fitted.
    [[nodiscard]] Forecast forecast(Eigen::Index horizon) const override;

    /// One-step-ahead in-sample fitted values (fitted[t] is the forecast of y[t]).
    [[nodiscard]] Eigen::VectorXd fitted_values() const override;

    /// Sum of squared one-step-ahead errors on the training data.
    [[nodiscard]] double sse() const noexcept { return sse_; }

    [[nodiscard]] double alpha() const noexcept { return parameters_->alpha; }

private:
    double level_ = 0.0;
    double sse_ = 0.0;
    Eigen::VectorXd in_sample_;  // one-step-ahead fitted values
};

} // namespace nabia::forecast
