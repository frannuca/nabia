//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_IFORECASTER_H
#define NABIA_IFORECASTER_H
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <Eigen/Dense>

/// Probabilistic forecast for steps h = 1..H after the end of the series.
/// The predictive distribution of each step is summarized by its mean and
/// variance (Gaussian for the models in nabia).
struct Forecast {
    Eigen::VectorXd mean;      // point forecast of y[T + h - 1], h = 1..H
    Eigen::VectorXd variance;  // predictive variance of the observation (noise included)

    [[nodiscard]] Eigen::Index horizon() const { return mean.size(); }
    [[nodiscard]] Eigen::VectorXd stddev() const { return variance.cwiseMax(0.0).cwiseSqrt(); }

    /// Central Gaussian prediction interval with coverage `level` in (0, 1),
    /// e.g. 0.95: mean -+ z * stddev with z = Phi^-1((1 + level) / 2).
    [[nodiscard]] Eigen::VectorXd lower(double level) const;
    [[nodiscard]] Eigen::VectorXd upper(double level) const;
};

/// Base of every nabia forecasting model, parameterized by its parameter struct T.
///
/// Models work on a univariate, equally spaced series y[0..T-1] (one value per
/// time step). NaN entries are treated as missing where a model supports it
/// (KalmanFilter); other models reject them.
///   fit(y)          estimates the model state (and, for some models, hyperparameters);
///   forecast(H)     predicts the next H steps, with uncertainty;
///   fitted_values() returns the in-sample fit, one value per observation.
template<typename T>
class IForecaster {
public:
    using Params = T;

    explicit IForecaster(std::unique_ptr<T> parameters) : parameters_(std::move(parameters)) {
        if (!parameters_) {
            throw std::invalid_argument("IForecaster: parameters are null");
        }
    }
    virtual ~IForecaster() = default;

    IForecaster(const IForecaster&) = delete;
    IForecaster& operator=(const IForecaster&) = delete;
    IForecaster(IForecaster&&) noexcept = default;
    IForecaster& operator=(IForecaster&&) noexcept = default;

    /// Fits the model to `y`. Throws std::invalid_argument on an unusable series.
    virtual void fit(const Eigen::VectorXd& y) = 0;

    /// Forecasts `horizon` >= 0 steps ahead. Throws std::logic_error before fit().
    [[nodiscard]] virtual Forecast forecast(Eigen::Index horizon) const = 0;

    /// In-sample fit: one-step-ahead predictions (SES, KalmanFilter) or the
    /// posterior mean at the training times (GaussianProcess).
    [[nodiscard]] virtual Eigen::VectorXd fitted_values() const = 0;

    [[nodiscard]] bool is_fitted() const { return fitted_; }
    [[nodiscard]] const T& parameters() const { return *parameters_; }

protected:
    void require_fitted(const char* who) const {
        if (!fitted_) {
            throw std::logic_error(std::string(who) + ": model is not fitted");
        }
    }
    static void require_horizon(Eigen::Index horizon, const char* who) {
        if (horizon < 0) {
            throw std::invalid_argument(std::string(who) + ": horizon must be >= 0");
        }
    }

    std::unique_ptr<T> parameters_;
    bool fitted_ = false;
};

#endif // NABIA_IFORECASTER_H
