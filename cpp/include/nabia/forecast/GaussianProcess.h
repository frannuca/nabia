//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_GAUSSIANPROCESS_H
#define NABIA_GAUSSIANPROCESS_H
#include "nabia/forecast/IForecaster.h"
#include "nabia/forecast/Kernels.h"
#include <memory>
#include <Eigen/Dense>

struct GaussianProcessParams {
    /// Prior covariance over time (time unit = one step of the series). The
    /// kernel's hyperparameters are the starting point when they are optimized.
    std::unique_ptr<IKernel> kernel;
    double noise_variance = 1e-2;          // observation noise sigma_n^2 (in normalized units if normalize)
    bool normalize = true;                 // fit on (y - mean) / std, so unit-scale hyperparameters are sensible
    bool optimize_hyperparameters = true;  // maximize the log marginal likelihood over kernel and noise
    bool optimize_noise = true;            // include sigma_n^2 in the optimization
    double min_noise_variance = 1e-8;      // floor on sigma_n^2 (numerical jitter)
    int max_iter = 200;                    // L-BFGS iterations for the hyperparameters
};

/// Gaussian process regression on the time index, used as a forecaster.
///
/// Model: y[t] = m + f(t) + e[t], with f ~ GP(0, k), e ~ N(0, sigma_n^2), and
/// m the sample mean. fit() places the training points at t = 0..T-1 and,
/// unless disabled, chooses the kernel hyperparameters and the noise by
/// maximizing the log marginal likelihood
///     log p(y) = -1/2 y'K^-1 y - 1/2 log|K| - T/2 log(2 pi),   K = K_f + sigma_n^2 I,
/// with L-BFGS on log-parameters and analytic gradients. Prediction at new
/// times uses the Cholesky factor of K (O(T^3) to fit, O(T^2) per point).
///
/// Far beyond the data the forecast returns to the mean with the prior
/// variance, at a rate set by the length scales. Use a PeriodicKernel (or a
/// sum with one) to extrapolate seasonality.
class GaussianProcess: public IForecaster<GaussianProcessParams> {
public:
    /// Throws std::invalid_argument without a kernel or with a non-positive noise variance.
    explicit GaussianProcess(std::unique_ptr<GaussianProcessParams> parameters);

    /// Throws std::invalid_argument if y has fewer than 2 points or non-finite values.
    void fit(const Eigen::VectorXd& y) override;
    /// Predictions at t = T..T+horizon-1, including the observation noise.
    [[nodiscard]] Forecast forecast(Eigen::Index horizon) const override;
    /// Posterior mean of m + f at the training times.
    [[nodiscard]] Eigen::VectorXd fitted_values() const override;

    /// Predictions of y at arbitrary times (interpolation or extrapolation).
    [[nodiscard]] Forecast predict(const Eigen::VectorXd& times) const;

    /// The kernel with the fitted hyperparameters (in normalized units if normalize).
    [[nodiscard]] const IKernel& kernel() const;
    /// Fitted noise variance sigma_n^2 (in normalized units if normalize).
    [[nodiscard]] double noise_variance() const;
    [[nodiscard]] double log_marginal_likelihood() const;

private:
    std::unique_ptr<IKernel> kernel_;  // fitted copy; the parameters keep the initial kernel
    double noise_ = 0.0;
    double offset_ = 0.0;              // mean subtracted from y
    double scale_ = 1.0;               // std dividing y
    Eigen::VectorXd times_;
    Eigen::MatrixXd chol_;             // lower Cholesky factor L of K
    Eigen::VectorXd alpha_;            // K^-1 z
    double log_ml_ = 0.0;
};

#endif // NABIA_GAUSSIANPROCESS_H
