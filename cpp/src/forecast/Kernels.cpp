//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/forecast/Kernels.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

using Eigen::Index;
using Eigen::VectorXd;

namespace {

void require_positive(double v, const char* what) {
    if (!(v > 0.0 && std::isfinite(v))) {
        throw std::invalid_argument(std::string("kernel: ") + what + " must be positive");
    }
}

void require_size(const VectorXd& v, Index n) {
    if (v.size() != n) {
        throw std::invalid_argument("kernel: wrong number of hyperparameters");
    }
}

} // namespace

Eigen::MatrixXd IKernel::matrix(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
    Eigen::MatrixXd K(a.size(), b.size());
    for (Index j = 0; j < b.size(); ++j) {
        for (Index i = 0; i < a.size(); ++i) {
            K(i, j) = (*this)(a(i), b(j));
        }
    }
    return K;
}

// --- Squared exponential ------------------------------------------------------

SquaredExponentialKernel::SquaredExponentialKernel(double variance, double length_scale)
    : variance_(variance), length_(length_scale) {
    require_positive(variance, "variance");
    require_positive(length_scale, "length scale");
}

double SquaredExponentialKernel::operator()(double t1, double t2) const {
    const double r = (t1 - t2) / length_;
    return variance_ * std::exp(-0.5 * r * r);
}

Eigen::VectorXd SquaredExponentialKernel::log_hyperparameters() const {
    return Eigen::Vector2d(std::log(variance_), std::log(length_));
}

void SquaredExponentialKernel::set_log_hyperparameters(const Eigen::VectorXd& log_theta) {
    require_size(log_theta, 2);
    variance_ = std::exp(log_theta(0));
    length_ = std::exp(log_theta(1));
}

Eigen::VectorXd SquaredExponentialKernel::gradient(double t1, double t2) const {
    const double r = (t1 - t2) / length_;
    const double k = variance_ * std::exp(-0.5 * r * r);
    return Eigen::Vector2d(k, k * r * r);  // d/dlog(variance), d/dlog(length)
}

std::unique_ptr<IKernel> SquaredExponentialKernel::clone() const {
    return std::make_unique<SquaredExponentialKernel>(*this);
}

// --- Matern 5/2 ----------------------------------------------------------------

Matern52Kernel::Matern52Kernel(double variance, double length_scale) : variance_(variance), length_(length_scale) {
    require_positive(variance, "variance");
    require_positive(length_scale, "length scale");
}

double Matern52Kernel::operator()(double t1, double t2) const {
    const double s = std::sqrt(5.0) * std::abs(t1 - t2) / length_;
    return variance_ * (1.0 + s + s * s / 3.0) * std::exp(-s);
}

Eigen::VectorXd Matern52Kernel::log_hyperparameters() const {
    return Eigen::Vector2d(std::log(variance_), std::log(length_));
}

void Matern52Kernel::set_log_hyperparameters(const Eigen::VectorXd& log_theta) {
    require_size(log_theta, 2);
    variance_ = std::exp(log_theta(0));
    length_ = std::exp(log_theta(1));
}

Eigen::VectorXd Matern52Kernel::gradient(double t1, double t2) const {
    const double s = std::sqrt(5.0) * std::abs(t1 - t2) / length_;
    const double e = std::exp(-s);
    const double k = variance_ * (1.0 + s + s * s / 3.0) * e;
    // dk/ds = -variance e s (1 + s) / 3 and ds/dlog(length) = -s.
    return Eigen::Vector2d(k, variance_ * e * s * s * (1.0 + s) / 3.0);
}

std::unique_ptr<IKernel> Matern52Kernel::clone() const {
    return std::make_unique<Matern52Kernel>(*this);
}

// --- Periodic ------------------------------------------------------------------

PeriodicKernel::PeriodicKernel(double variance, double length_scale, double period)
    : variance_(variance), length_(length_scale), period_(period) {
    require_positive(variance, "variance");
    require_positive(length_scale, "length scale");
    require_positive(period, "period");
}

double PeriodicKernel::operator()(double t1, double t2) const {
    const double s = std::sin(std::numbers::pi * (t1 - t2) / period_);
    return variance_ * std::exp(-2.0 * s * s / (length_ * length_));
}

Eigen::VectorXd PeriodicKernel::log_hyperparameters() const {
    return Eigen::Vector3d(std::log(variance_), std::log(length_), std::log(period_));
}

void PeriodicKernel::set_log_hyperparameters(const Eigen::VectorXd& log_theta) {
    require_size(log_theta, 3);
    variance_ = std::exp(log_theta(0));
    length_ = std::exp(log_theta(1));
    period_ = std::exp(log_theta(2));
}

Eigen::VectorXd PeriodicKernel::gradient(double t1, double t2) const {
    const double u = std::numbers::pi * (t1 - t2) / period_;
    const double s = std::sin(u);
    const double l2 = length_ * length_;
    const double k = variance_ * std::exp(-2.0 * s * s / l2);
    // d sin^2(u) / dlog(period) = -u sin(2u)
    return Eigen::Vector3d(k, k * 4.0 * s * s / l2, k * 2.0 * u * std::sin(2.0 * u) / l2);
}

std::unique_ptr<IKernel> PeriodicKernel::clone() const {
    return std::make_unique<PeriodicKernel>(*this);
}

// --- Sum and product -------------------------------------------------------------

namespace {

void require_kernels(const std::unique_ptr<IKernel>& a, const std::unique_ptr<IKernel>& b) {
    if (!a || !b) {
        throw std::invalid_argument("kernel: composite kernels need two non-null kernels");
    }
}

} // namespace

SumKernel::SumKernel(std::unique_ptr<IKernel> k1, std::unique_ptr<IKernel> k2) : k1_(std::move(k1)), k2_(std::move(k2)) {
    require_kernels(k1_, k2_);
}

double SumKernel::operator()(double t1, double t2) const {
    return (*k1_)(t1, t2) + (*k2_)(t1, t2);
}

Eigen::Index SumKernel::num_hyperparameters() const {
    return k1_->num_hyperparameters() + k2_->num_hyperparameters();
}

Eigen::VectorXd SumKernel::log_hyperparameters() const {
    VectorXd theta(num_hyperparameters());
    theta << k1_->log_hyperparameters(), k2_->log_hyperparameters();
    return theta;
}

void SumKernel::set_log_hyperparameters(const Eigen::VectorXd& log_theta) {
    require_size(log_theta, num_hyperparameters());
    const Index n1 = k1_->num_hyperparameters();
    k1_->set_log_hyperparameters(log_theta.head(n1));
    k2_->set_log_hyperparameters(log_theta.tail(log_theta.size() - n1));
}

Eigen::VectorXd SumKernel::gradient(double t1, double t2) const {
    VectorXd g(num_hyperparameters());
    g << k1_->gradient(t1, t2), k2_->gradient(t1, t2);
    return g;
}

std::unique_ptr<IKernel> SumKernel::clone() const {
    return std::make_unique<SumKernel>(k1_->clone(), k2_->clone());
}

ProductKernel::ProductKernel(std::unique_ptr<IKernel> k1, std::unique_ptr<IKernel> k2)
    : k1_(std::move(k1)), k2_(std::move(k2)) {
    require_kernels(k1_, k2_);
}

double ProductKernel::operator()(double t1, double t2) const {
    return (*k1_)(t1, t2) * (*k2_)(t1, t2);
}

Eigen::Index ProductKernel::num_hyperparameters() const {
    return k1_->num_hyperparameters() + k2_->num_hyperparameters();
}

Eigen::VectorXd ProductKernel::log_hyperparameters() const {
    VectorXd theta(num_hyperparameters());
    theta << k1_->log_hyperparameters(), k2_->log_hyperparameters();
    return theta;
}

void ProductKernel::set_log_hyperparameters(const Eigen::VectorXd& log_theta) {
    require_size(log_theta, num_hyperparameters());
    const Index n1 = k1_->num_hyperparameters();
    k1_->set_log_hyperparameters(log_theta.head(n1));
    k2_->set_log_hyperparameters(log_theta.tail(log_theta.size() - n1));
}

Eigen::VectorXd ProductKernel::gradient(double t1, double t2) const {
    VectorXd g(num_hyperparameters());
    g << (*k2_)(t1, t2) * k1_->gradient(t1, t2), (*k1_)(t1, t2) * k2_->gradient(t1, t2);
    return g;
}

std::unique_ptr<IKernel> ProductKernel::clone() const {
    return std::make_unique<ProductKernel>(k1_->clone(), k2_->clone());
}
