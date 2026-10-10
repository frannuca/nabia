//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_KALMANFILTER_H
#define NABIA_KALMANFILTER_H
#include "nabia/forecast/IForecaster.h"
#include <memory>
#include <vector>
#include <Eigen/Dense>

/// Linear Gaussian state-space model with n states and a scalar observation:
///     x[t] = F x[t-1] + w[t],   w ~ N(0, Q)
///     y[t] = H x[t]   + v[t],   v ~ N(0, R)
/// with prior x[0] ~ N(x0, P0).
struct KalmanFilterParams {
    Eigen::MatrixXd transition;          // F, n x n
    Eigen::RowVectorXd observation;      // H, 1 x n
    Eigen::MatrixXd process_noise;       // Q, n x n, symmetric positive semi-definite
    double observation_noise = 1.0;      // R > 0
    Eigen::VectorXd initial_state;       // x0, n
    Eigen::MatrixXd initial_covariance;  // P0, n x n, symmetric positive semi-definite

    /// Random walk plus noise: level[t] = level[t-1] + w, y[t] = level[t] + v.
    /// A large initial variance makes the prior nearly uninformative (diffuse).
    [[nodiscard]] static KalmanFilterParams local_level(double level_variance, double observation_variance,
                                                        double initial_level = 0.0, double initial_variance = 1e7);

    /// Level and slope: level[t] = level[t-1] + slope[t-1] + w1, slope[t] = slope[t-1] + w2,
    /// y[t] = level[t] + v. Forecasts extrapolate the current trend.
    [[nodiscard]] static KalmanFilterParams local_linear_trend(double level_variance, double slope_variance,
                                                               double observation_variance,
                                                               double initial_variance = 1e7);
};

/// Kalman filter forecaster for KalmanFilterParams models.
///
/// fit() runs the filter over the series: at each step it predicts the state,
/// then corrects it with the observation (Joseph-form covariance update, which
/// keeps P symmetric positive semi-definite). NaN observations are treated as
/// missing: the step is predicted but not corrected. forecast() propagates the
/// last filtered state through the transition; the variance grows with the
/// horizon. smoothed_states() runs the Rauch-Tung-Striebel smoother.
class KalmanFilter: public IForecaster<KalmanFilterParams> {
public:
    /// Throws std::invalid_argument on inconsistent dimensions, R <= 0, or
    /// non-symmetric Q or P0.
    explicit KalmanFilter(std::unique_ptr<KalmanFilterParams> parameters);

    void fit(const Eigen::VectorXd& y) override;
    [[nodiscard]] Forecast forecast(Eigen::Index horizon) const override;
    /// One-step-ahead predictions H x[t|t-1].
    [[nodiscard]] Eigen::VectorXd fitted_values() const override;

    [[nodiscard]] Eigen::Index state_dim() const { return parameters_->transition.rows(); }
    /// Filtered states x[t|t] as columns (n x T).
    [[nodiscard]] const Eigen::MatrixXd& filtered_states() const;
    /// Filtered covariances P[t|t].
    [[nodiscard]] const std::vector<Eigen::MatrixXd>& filtered_covariances() const;
    /// Innovations y[t] - H x[t|t-1] (NaN where y is missing) and their variances.
    [[nodiscard]] const Eigen::VectorXd& innovations() const;
    [[nodiscard]] const Eigen::VectorXd& innovation_variances() const;
    /// Gaussian log-likelihood of the observed values (prediction error decomposition).
    [[nodiscard]] double log_likelihood() const;
    /// Smoothed states x[t|T] as columns (n x T): uses the whole series.
    [[nodiscard]] Eigen::MatrixXd smoothed_states() const;

private:
    Eigen::MatrixXd filtered_states_;
    std::vector<Eigen::MatrixXd> filtered_covariances_;
    Eigen::MatrixXd predicted_states_;               // x[t|t-1]
    std::vector<Eigen::MatrixXd> predicted_covariances_;  // P[t|t-1]
    Eigen::VectorXd one_step_;
    Eigen::VectorXd innovations_;
    Eigen::VectorXd innovation_variances_;
    double log_likelihood_ = 0.0;
};

#endif // NABIA_KALMANFILTER_H
