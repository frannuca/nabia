//
// Created by Francisco Nunez on 24.10.2026.
//

#include "nabia/regression/ElasticNet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace nabia::regression {

namespace {

constexpr double kMinAlphaForLambdaMax = 1e-3;  // glmnet's floor, so ridge gets a finite path

/// The design the coordinate descent works on: centred (with an intercept)
/// and scaled to unit population variance (with standardize).
struct Standardized {
    MatrixXd X;           // n x p working design
    VectorXd y;           // centred target with an intercept
    VectorXd x_mean;      // subtracted from each column (0 without intercept)
    VectorXd x_scale;     // dividing each column; 0 marks a constant column (coefficient fixed at 0)
    VectorXd v;           // ||x_j||^2 / n of the working columns (1 if standardized, 0 if constant)
    double y_mean = 0.0;
    double y_var = 0.0;   // ||y||^2 / n of the working target
};

void validate_data(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("ElasticNet: X must have one row per observation of y");
    }
    if (X.rows() == 0) {
        throw std::invalid_argument("ElasticNet: no observations");
    }
    if (!X.allFinite() || !y.allFinite()) {
        throw std::invalid_argument("ElasticNet: data must be finite");
    }
}

Standardized standardize_data(const MatrixXd& X, const VectorXd& y, bool fit_intercept, bool standardize) {
    const Index p = X.cols();
    const auto n = static_cast<double>(X.rows());
    Standardized s;
    s.x_mean = fit_intercept ? VectorXd(X.colwise().mean().transpose()) : VectorXd::Zero(p);
    s.y_mean = fit_intercept ? y.mean() : 0.0;
    s.X = X.rowwise() - s.x_mean.transpose();
    s.y = y.array() - s.y_mean;
    s.y_var = s.y.squaredNorm() / n;
    s.x_scale.resize(p);
    s.v.resize(p);
    for (Index j = 0; j < p; ++j) {
        const double ms = s.X.col(j).squaredNorm() / n;
        // Relative to the raw magnitude: centring a constant column leaves rounding noise.
        const double raw = X.col(j).squaredNorm() / n;
        if (ms <= 1e-24 * std::max(1.0, raw)) {
            s.X.col(j).setZero();
            s.x_scale(j) = 0.0;
            s.v(j) = 0.0;
        } else if (standardize) {
            s.x_scale(j) = std::sqrt(ms);
            s.X.col(j) /= s.x_scale(j);
            s.v(j) = 1.0;
        } else {
            s.x_scale(j) = 1.0;
            s.v(j) = ms;
        }
    }
    return s;
}

double soft_threshold(double z, double t) {
    if (z > t) {
        return z - t;
    }
    if (z < -t) {
        return z + t;
    }
    return 0.0;
}

/// Coordinate descent state on one standardized design, reused along a path.
class CoordinateDescent {
public:
    explicit CoordinateDescent(const Standardized& s)
        : s_(s), beta_(VectorXd::Zero(s.X.cols())), residual_(s.y) {}

    /// Solves at (lambda, alpha) starting from the current coefficients.
    /// Returns the sweeps used; `converged` reports whether tolerance was met.
    int solve(double lambda, double alpha, double tolerance, int max_iter, bool& converged) {
        l1_ = lambda * alpha;
        l2_ = lambda * (1.0 - alpha);
        const double threshold = tolerance * std::max(s_.y_var, std::numeric_limits<double>::min());
        int sweeps = 0;
        converged = false;
        while (sweeps < max_iter) {
            ++sweeps;
            if (sweep(false) < threshold) {
                converged = true;
                break;
            }
            while (sweeps < max_iter) {
                ++sweeps;
                if (sweep(true) < threshold) {
                    break;
                }
            }
        }
        return sweeps;
    }

    [[nodiscard]] const VectorXd& beta() const { return beta_; }

private:
    /// One pass over the factors (only the non-zero ones if active_only);
    /// returns the largest v_j (delta b_j)^2.
    double sweep(bool active_only) {
        const auto n = static_cast<double>(s_.X.rows());
        double max_change = 0.0;
        for (Index j = 0; j < beta_.size(); ++j) {
            const double vj = s_.v(j);
            if (vj == 0.0 || (active_only && beta_(j) == 0.0)) {
                continue;
            }
            const double old = beta_(j);
            const double z = s_.X.col(j).dot(residual_) / n + vj * old;
            const double updated = soft_threshold(z, l1_) / (vj + l2_);
            const double delta = updated - old;
            if (delta != 0.0) {
                residual_ -= delta * s_.X.col(j);
                beta_(j) = updated;
                max_change = std::max(max_change, vj * delta * delta);
            }
        }
        return max_change;
    }

    const Standardized& s_;
    VectorXd beta_;      // coefficients of the working (standardized) design
    VectorXd residual_;  // y - X beta in working units
    double l1_ = 0.0;
    double l2_ = 0.0;
};

/// tr(X_A (X_A'X_A + n l2 I)^-1 X_A') over the active set A, plus the intercept.
double degrees_of_freedom(const Standardized& s, const VectorXd& beta, double l2, bool fit_intercept) {
    std::vector<Index> active;
    for (Index j = 0; j < beta.size(); ++j) {
        if (beta(j) != 0.0) {
            active.push_back(j);
        }
    }
    double df = fit_intercept ? 1.0 : 0.0;
    if (active.empty()) {
        return df;
    }
    const MatrixXd Xa = s.X(Eigen::placeholders::all, active);
    const Eigen::SelfAdjointEigenSolver<MatrixXd> eig(Xa.transpose() * Xa, Eigen::EigenvaluesOnly);
    const VectorXd& d = eig.eigenvalues();
    const double ridge = static_cast<double>(s.X.rows()) * l2;
    const double floor = 1e-10 * std::max(d.maxCoeff(), 0.0);
    for (Index i = 0; i < d.size(); ++i) {
        if (d(i) > floor) {
            df += d(i) / (d(i) + ridge);
        }
    }
    return df;
}

ElasticNetFit make_fit(const Standardized& s, const VectorXd& beta, double lambda, double alpha,
                       bool fit_intercept) {
    ElasticNetFit f;
    f.lambda = lambda;
    f.alpha = alpha;
    f.coefficients = VectorXd::Zero(beta.size());
    for (Index j = 0; j < beta.size(); ++j) {
        if (s.x_scale(j) > 0.0) {
            f.coefficients(j) = beta(j) / s.x_scale(j);
        }
    }
    f.intercept = fit_intercept ? s.y_mean - s.x_mean.dot(f.coefficients) : 0.0;
    f.nonzero = (beta.array() != 0.0).count();
    f.degrees_of_freedom = degrees_of_freedom(s, beta, lambda * (1.0 - alpha), fit_intercept);
    return f;
}

} // namespace

void validate(const ElasticNetParams& params) {
    if (!(params.lambda >= 0.0) || !std::isfinite(params.lambda)) {
        throw std::invalid_argument("ElasticNet: lambda must be finite and >= 0");
    }
    if (!(params.alpha >= 0.0 && params.alpha <= 1.0)) {
        throw std::invalid_argument("ElasticNet: alpha must be in [0, 1]");
    }
    if (!(params.tolerance > 0.0)) {
        throw std::invalid_argument("ElasticNet: tolerance must be > 0");
    }
    if (params.max_iter < 1) {
        throw std::invalid_argument("ElasticNet: max_iter must be >= 1");
    }
}

double lambda_max(const MatrixXd& X, const VectorXd& y, double alpha, bool fit_intercept, bool standardize) {
    validate_data(X, y);
    if (!(alpha >= 0.0 && alpha <= 1.0)) {
        throw std::invalid_argument("lambda_max: alpha must be in [0, 1]");
    }
    if (X.cols() == 0) {
        return 0.0;
    }
    const Standardized s = standardize_data(X, y, fit_intercept, standardize);
    // Same arithmetic as the first coordinate update, plus a few ulps so that
    // lambda * alpha >= |gradient| survives the rounding of the division.
    double gradient = 0.0;
    for (Index j = 0; j < X.cols(); ++j) {
        gradient = std::max(gradient, std::abs(s.X.col(j).dot(s.y) / static_cast<double>(X.rows())));
    }
    return gradient / std::max(alpha, kMinAlphaForLambdaMax) * (1.0 + 1e-12);
}

std::vector<double> lambda_grid(double lambda_max, Index count, double min_ratio) {
    if (count < 1) {
        throw std::invalid_argument("lambda_grid: count must be >= 1");
    }
    if (!(min_ratio > 0.0 && min_ratio <= 1.0)) {
        throw std::invalid_argument("lambda_grid: min_ratio must be in (0, 1]");
    }
    if (!(lambda_max >= 0.0) || !std::isfinite(lambda_max)) {
        throw std::invalid_argument("lambda_grid: lambda_max must be finite and >= 0");
    }
    std::vector<double> grid(static_cast<std::size_t>(count), 0.0);
    if (lambda_max == 0.0) {
        return grid;
    }
    const double log_ratio = std::log(min_ratio);
    for (Index i = 0; i < count; ++i) {
        const double t = count == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(count - 1);
        grid[static_cast<std::size_t>(i)] = lambda_max * std::exp(t * log_ratio);
    }
    return grid;
}

std::vector<ElasticNetFit> elastic_net_path(const MatrixXd& X, const VectorXd& y, const ElasticNetParams& params,
                                            const std::vector<double>& lambdas) {
    ElasticNetParams checked = params;
    checked.lambda = 0.0;  // ignored here
    validate(checked);
    validate_data(X, y);
    for (double l : lambdas) {
        if (!(l >= 0.0) || !std::isfinite(l)) {
            throw std::invalid_argument("elastic_net_path: lambdas must be finite and >= 0");
        }
    }
    const Standardized s = standardize_data(X, y, params.fit_intercept, params.standardize);
    CoordinateDescent cd(s);
    std::vector<ElasticNetFit> path;
    path.reserve(lambdas.size());
    for (double l : lambdas) {
        bool converged = false;
        const int sweeps = cd.solve(l, params.alpha, params.tolerance, params.max_iter, converged);
        ElasticNetFit f = make_fit(s, cd.beta(), l, params.alpha, params.fit_intercept);
        f.iterations = sweeps;
        f.converged = converged;
        path.push_back(std::move(f));
    }
    return path;
}

} // namespace nabia::regression

VectorXd ElasticNetFit::predict(const MatrixXd& X) const {
    if (X.cols() != coefficients.size()) {
        throw std::invalid_argument("ElasticNetFit::predict: X has " + std::to_string(X.cols()) +
                                    " columns, the model " + std::to_string(coefficients.size()));
    }
    return (X * coefficients).array() + intercept;
}

ElasticNet::ElasticNet(std::unique_ptr<ElasticNetParams> parameters) : parameters_(std::move(parameters)) {
    if (!parameters_) {
        throw std::invalid_argument("ElasticNet: parameters are null");
    }
    nabia::regression::validate(*parameters_);
}

void ElasticNet::fit(const MatrixXd& X, const VectorXd& y) {
    fit_ = nabia::regression::elastic_net_path(X, y, *parameters_, {parameters_->lambda}).front();
    fitted_ = true;
}

VectorXd ElasticNet::predict(const MatrixXd& X) const { return result().predict(X); }

const ElasticNetFit& ElasticNet::result() const {
    if (!fitted_) {
        throw std::logic_error("ElasticNet: call fit() first");
    }
    return fit_;
}
