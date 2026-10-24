//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/forecast/GaussianProcess.h"

#include "nabia/optimization/convex/LBFGS.h"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

constexpr double kLog2Pi = 1.8378770664093453;
constexpr double kMaxLogParam = 20.0;  // keep exp(log theta) finite and the Gram matrix usable

struct Factorization {
    Eigen::LLT<MatrixXd> llt;
    VectorXd alpha;      // K^-1 z
    double log_ml = 0.0; // log marginal likelihood
};

/// Cholesky of K = K_f + noise I and the log marginal likelihood; nullopt if K is not positive definite.
std::optional<Factorization> factorize(const MatrixXd& Kf, double noise, const VectorXd& z) {
    MatrixXd K = Kf;
    K.diagonal().array() += noise;
    Factorization f{Eigen::LLT<MatrixXd>(K), VectorXd(), 0.0};
    if (f.llt.info() != Eigen::Success) {
        return std::nullopt;
    }
    f.alpha = f.llt.solve(z);
    const double log_det = 2.0 * f.llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    f.log_ml = -0.5 * z.dot(f.alpha) - 0.5 * log_det - 0.5 * static_cast<double>(z.size()) * kLog2Pi;
    return f;
}

} // namespace

GaussianProcess::GaussianProcess(std::unique_ptr<GaussianProcessParams> parameters) : IForecaster(std::move(parameters)) {
    const GaussianProcessParams& p = *parameters_;
    if (!p.kernel) {
        throw std::invalid_argument("GaussianProcess: a kernel is required");
    }
    if (!(p.noise_variance > 0.0) || !(p.min_noise_variance > 0.0)) {
        throw std::invalid_argument("GaussianProcess: noise variances must be positive");
    }
}

void GaussianProcess::fit(const Eigen::VectorXd& y) {
    const GaussianProcessParams& p = *parameters_;
    const Index T = y.size();
    if (T < 2 || !y.allFinite()) {
        throw std::invalid_argument("GaussianProcess::fit: need at least 2 finite observations");
    }

    times_ = VectorXd::LinSpaced(T, 0.0, static_cast<double>(T - 1));
    offset_ = y.mean();
    scale_ = 1.0;
    if (p.normalize) {
        const double sd = std::sqrt((y.array() - offset_).square().sum() / static_cast<double>(T - 1));
        scale_ = sd > 0.0 ? sd : 1.0;
    }
    const VectorXd z = (y.array() - offset_) / scale_;

    kernel_ = p.kernel->clone();
    noise_ = std::max(p.noise_variance, p.min_noise_variance);
    const Index nk = kernel_->num_hyperparameters();

    if (p.optimize_hyperparameters) {
        // theta = [log kernel hyperparameters, log noise (if optimized)]; minimize -log p(z).
        const Index nt = nk + (p.optimize_noise ? 1 : 0);
        VectorXd theta0(nt);
        theta0.head(nk) = kernel_->log_hyperparameters();
        if (p.optimize_noise) {
            theta0(nk) = std::log(noise_);
        }
        const std::unique_ptr<IKernel> work = kernel_->clone();

        const LBFGS::ValueAndGradient nll = [&](const VectorXd& theta, VectorXd& grad) {
            constexpr double inf = std::numeric_limits<double>::infinity();
            if ((theta.array().abs() > kMaxLogParam).any()) {
                grad.setZero();
                return inf;
            }
            work->set_log_hyperparameters(theta.head(nk));
            const double noise = p.optimize_noise ? std::max(std::exp(theta(nk)), p.min_noise_variance) : noise_;
            const MatrixXd Kf = work->matrix(times_, times_);
            const auto f = factorize(Kf, noise, z);
            if (!f) {
                grad.setZero();
                return inf;
            }
            // d(-log p)/d theta_j = -1/2 tr((alpha alpha' - K^-1) dK/d theta_j)
            const MatrixXd W = f->alpha * f->alpha.transpose() - f->llt.solve(MatrixXd::Identity(T, T));
            grad.setZero();
            for (Index j = 0; j < T; ++j) {
                for (Index i = 0; i < T; ++i) {
                    grad.head(nk) -= 0.5 * W(i, j) * work->gradient(times_(i), times_(j));
                }
            }
            if (p.optimize_noise) {
                grad(nk) = -0.5 * noise * W.trace();  // dK/dlog(noise) = noise I
            }
            return -f->log_ml;
        };

        LBFGSParams lbfgs{nt};
        lbfgs.tolerance = 1e-5;
        lbfgs.max_iter = p.max_iter;
        LBFGS solver(std::make_unique<LBFGSParams>(lbfgs), nll);
        const VectorXd theta = solver.optimize(theta0);
        kernel_->set_log_hyperparameters(theta.head(nk));
        if (p.optimize_noise) {
            noise_ = std::max(std::exp(theta(nk)), p.min_noise_variance);
        }
    }

    auto f = factorize(kernel_->matrix(times_, times_), noise_, z);
    if (!f) {
        throw std::invalid_argument("GaussianProcess::fit: kernel matrix is not positive definite; increase the noise");
    }
    chol_ = f->llt.matrixL();
    alpha_ = std::move(f->alpha);
    log_ml_ = f->log_ml - static_cast<double>(T) * std::log(scale_);  // back to the units of y
    fitted_ = true;
}

Forecast GaussianProcess::predict(const Eigen::VectorXd& times) const {
    require_fitted("GaussianProcess::predict");
    const MatrixXd Ks = kernel_->matrix(times_, times);  // T x m
    const MatrixXd V = chol_.triangularView<Eigen::Lower>().solve(Ks);
    VectorXd prior(times.size());
    for (Index j = 0; j < times.size(); ++j) {
        prior(j) = (*kernel_)(times(j), times(j));
    }
    Forecast out;
    out.mean = (Ks.transpose() * alpha_).array() * scale_ + offset_;
    out.variance = ((prior - V.colwise().squaredNorm().transpose()).cwiseMax(0.0).array() + noise_) * scale_ * scale_;
    return out;
}

Forecast GaussianProcess::forecast(Eigen::Index horizon) const {
    require_fitted("GaussianProcess::forecast");
    require_horizon(horizon, "GaussianProcess::forecast");
    const double T = static_cast<double>(times_.size());
    return predict(VectorXd::LinSpaced(horizon, T, T + static_cast<double>(horizon) - 1.0));
}

Eigen::VectorXd GaussianProcess::fitted_values() const {
    require_fitted("GaussianProcess::fitted_values");
    return (kernel_->matrix(times_, times_) * alpha_).array() * scale_ + offset_;
}

const IKernel& GaussianProcess::kernel() const {
    require_fitted("GaussianProcess::kernel");
    return *kernel_;
}

double GaussianProcess::noise_variance() const {
    require_fitted("GaussianProcess::noise_variance");
    return noise_;
}

double GaussianProcess::log_marginal_likelihood() const {
    require_fitted("GaussianProcess::log_marginal_likelihood");
    return log_ml_;
}
