//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/forecast/KalmanFilter.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

bool is_symmetric(const MatrixXd& m) {
    const double scale = std::max(1.0, m.cwiseAbs().maxCoeff());
    return m.rows() == m.cols() && ((m - m.transpose()).cwiseAbs().array() <= 1e-10 * scale).all();
}

} // namespace

KalmanFilterParams KalmanFilterParams::local_level(double level_variance, double observation_variance,
                                                   double initial_level, double initial_variance) {
    KalmanFilterParams p;
    p.transition = MatrixXd::Ones(1, 1);
    p.observation = Eigen::RowVectorXd::Ones(1);
    p.process_noise = MatrixXd::Constant(1, 1, level_variance);
    p.observation_noise = observation_variance;
    p.initial_state = VectorXd::Constant(1, initial_level);
    p.initial_covariance = MatrixXd::Constant(1, 1, initial_variance);
    return p;
}

KalmanFilterParams KalmanFilterParams::local_linear_trend(double level_variance, double slope_variance,
                                                          double observation_variance, double initial_variance) {
    KalmanFilterParams p;
    p.transition.resize(2, 2);
    p.transition << 1.0, 1.0,
                    0.0, 1.0;
    p.observation.resize(2);
    p.observation << 1.0, 0.0;
    p.process_noise = Eigen::Vector2d(level_variance, slope_variance).asDiagonal();
    p.observation_noise = observation_variance;
    p.initial_state = VectorXd::Zero(2);
    p.initial_covariance = initial_variance * MatrixXd::Identity(2, 2);
    return p;
}

KalmanFilter::KalmanFilter(std::unique_ptr<KalmanFilterParams> parameters) : IForecaster(std::move(parameters)) {
    const KalmanFilterParams& p = *parameters_;
    const Index n = p.transition.rows();
    if (n == 0 || p.transition.cols() != n || p.observation.size() != n || p.process_noise.rows() != n ||
        p.initial_state.size() != n || p.initial_covariance.rows() != n) {
        throw std::invalid_argument("KalmanFilter: F, H, Q, x0 and P0 must all have the state dimension");
    }
    if (!is_symmetric(p.process_noise) || !is_symmetric(p.initial_covariance)) {
        throw std::invalid_argument("KalmanFilter: Q and P0 must be symmetric");
    }
    if (!(p.observation_noise > 0.0 && std::isfinite(p.observation_noise))) {
        throw std::invalid_argument("KalmanFilter: observation noise R must be positive");
    }
}

void KalmanFilter::fit(const Eigen::VectorXd& y) {
    const KalmanFilterParams& p = *parameters_;
    const Index n = state_dim();
    const Index T = y.size();
    if (T == 0) {
        throw std::invalid_argument("KalmanFilter::fit: series is empty");
    }
    const MatrixXd& F = p.transition;
    const Eigen::RowVectorXd& H = p.observation;
    const MatrixXd I = MatrixXd::Identity(n, n);
    constexpr double log_2pi = 1.8378770664093453;  // log(2 pi)

    filtered_states_.resize(n, T);
    predicted_states_.resize(n, T);
    filtered_covariances_.assign(static_cast<std::size_t>(T), MatrixXd());
    predicted_covariances_.assign(static_cast<std::size_t>(T), MatrixXd());
    one_step_.resize(T);
    innovations_.resize(T);
    innovation_variances_.resize(T);
    log_likelihood_ = 0.0;

    VectorXd x = p.initial_state;
    MatrixXd P = p.initial_covariance;
    for (Index t = 0; t < T; ++t) {
        if (t > 0) {  // predict; the prior already describes x[0]
            x = F * x;
            P = F * P * F.transpose() + p.process_noise;
        }
        const auto ts = static_cast<std::size_t>(t);
        predicted_states_.col(t) = x;
        predicted_covariances_[ts] = P;

        const double y_hat = H.dot(x);
        const VectorXd PHt = P * H.transpose();
        const double S = H.dot(PHt) + p.observation_noise;
        one_step_(t) = y_hat;
        innovation_variances_(t) = S;

        if (std::isfinite(y(t))) {  // correct
            const double e = y(t) - y_hat;
            const VectorXd K = PHt / S;
            x += K * e;
            const MatrixXd A = I - K * H;
            P = A * P * A.transpose() + p.observation_noise * K * K.transpose();
            innovations_(t) = e;
            log_likelihood_ -= 0.5 * (log_2pi + std::log(S) + e * e / S);
        } else {
            innovations_(t) = std::numeric_limits<double>::quiet_NaN();
        }
        filtered_states_.col(t) = x;
        filtered_covariances_[ts] = P;
    }
    fitted_ = true;
}

Forecast KalmanFilter::forecast(Eigen::Index horizon) const {
    require_fitted("KalmanFilter::forecast");
    require_horizon(horizon, "KalmanFilter::forecast");
    const KalmanFilterParams& p = *parameters_;
    VectorXd x = filtered_states_.rightCols(1);
    MatrixXd P = filtered_covariances_.back();
    Forecast out{VectorXd(horizon), VectorXd(horizon)};
    for (Index h = 0; h < horizon; ++h) {
        x = p.transition * x;
        P = p.transition * P * p.transition.transpose() + p.process_noise;
        out.mean(h) = p.observation.dot(x);
        out.variance(h) = (p.observation * P * p.observation.transpose()).value() + p.observation_noise;
    }
    return out;
}

Eigen::VectorXd KalmanFilter::fitted_values() const {
    require_fitted("KalmanFilter::fitted_values");
    return one_step_;
}

const Eigen::MatrixXd& KalmanFilter::filtered_states() const {
    require_fitted("KalmanFilter::filtered_states");
    return filtered_states_;
}

const std::vector<Eigen::MatrixXd>& KalmanFilter::filtered_covariances() const {
    require_fitted("KalmanFilter::filtered_covariances");
    return filtered_covariances_;
}

const Eigen::VectorXd& KalmanFilter::innovations() const {
    require_fitted("KalmanFilter::innovations");
    return innovations_;
}

const Eigen::VectorXd& KalmanFilter::innovation_variances() const {
    require_fitted("KalmanFilter::innovation_variances");
    return innovation_variances_;
}

double KalmanFilter::log_likelihood() const {
    require_fitted("KalmanFilter::log_likelihood");
    return log_likelihood_;
}

Eigen::MatrixXd KalmanFilter::smoothed_states() const {
    require_fitted("KalmanFilter::smoothed_states");
    const MatrixXd& F = parameters_->transition;
    const Index T = filtered_states_.cols();
    MatrixXd smoothed = filtered_states_;
    for (Index t = T - 2; t >= 0; --t) {
        const auto ts = static_cast<std::size_t>(t);
        const MatrixXd& P_f = filtered_covariances_[ts];
        const MatrixXd& P_p = predicted_covariances_[ts + 1];
        // Smoother gain C = P[t|t] F' P[t+1|t]^-1 (P symmetric, so solve instead of inverting).
        const MatrixXd C = P_p.ldlt().solve(F * P_f).transpose();
        smoothed.col(t) = filtered_states_.col(t) + C * (smoothed.col(t + 1) - predicted_states_.col(t + 1));
    }
    return smoothed;
}
