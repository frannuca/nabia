//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_KERNELS_H
#define NABIA_KERNELS_H
#include <memory>
#include <Eigen/Dense>

/// Covariance function k(t, t') of a Gaussian process over time.
///
/// Hyperparameters (variances, length scales, periods) are positive, so they
/// are exposed as logarithms: optimizing over log-parameters is unconstrained
/// and better scaled. gradient() returns dk / d(log theta_j).
class IKernel {
public:
    virtual ~IKernel() = default;

    [[nodiscard]] virtual double operator()(double t1, double t2) const = 0;
    [[nodiscard]] virtual Eigen::Index num_hyperparameters() const = 0;
    [[nodiscard]] virtual Eigen::VectorXd log_hyperparameters() const = 0;
    virtual void set_log_hyperparameters(const Eigen::VectorXd& log_theta) = 0;
    [[nodiscard]] virtual Eigen::VectorXd gradient(double t1, double t2) const = 0;
    [[nodiscard]] virtual std::unique_ptr<IKernel> clone() const = 0;

    /// Gram matrix K_ij = k(a_i, b_j).
    [[nodiscard]] Eigen::MatrixXd matrix(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;
};

/// k = variance * exp(-r^2 / (2 length^2)), r = |t - t'|. Smooth (infinitely differentiable) functions.
class SquaredExponentialKernel: public IKernel {
public:
    /// Throws std::invalid_argument unless variance > 0 and length_scale > 0.
    SquaredExponentialKernel(double variance, double length_scale);
    [[nodiscard]] double operator()(double t1, double t2) const override;
    [[nodiscard]] Eigen::Index num_hyperparameters() const override { return 2; }
    [[nodiscard]] Eigen::VectorXd log_hyperparameters() const override;  // [log variance, log length]
    void set_log_hyperparameters(const Eigen::VectorXd& log_theta) override;
    [[nodiscard]] Eigen::VectorXd gradient(double t1, double t2) const override;
    [[nodiscard]] std::unique_ptr<IKernel> clone() const override;
    [[nodiscard]] double variance() const { return variance_; }
    [[nodiscard]] double length_scale() const { return length_; }

private:
    double variance_;
    double length_;
};

/// Matern 5/2: k = variance (1 + s + s^2 / 3) exp(-s), s = sqrt(5) r / length.
/// Twice-differentiable functions: rougher than the squared exponential, often more realistic.
class Matern52Kernel: public IKernel {
public:
    Matern52Kernel(double variance, double length_scale);
    [[nodiscard]] double operator()(double t1, double t2) const override;
    [[nodiscard]] Eigen::Index num_hyperparameters() const override { return 2; }
    [[nodiscard]] Eigen::VectorXd log_hyperparameters() const override;  // [log variance, log length]
    void set_log_hyperparameters(const Eigen::VectorXd& log_theta) override;
    [[nodiscard]] Eigen::VectorXd gradient(double t1, double t2) const override;
    [[nodiscard]] std::unique_ptr<IKernel> clone() const override;
    [[nodiscard]] double variance() const { return variance_; }
    [[nodiscard]] double length_scale() const { return length_; }

private:
    double variance_;
    double length_;
};

/// Periodic (exp-sine-squared): k = variance exp(-2 sin^2(pi r / period) / length^2).
/// Repeating patterns (seasonality) with the given period, in time steps.
class PeriodicKernel: public IKernel {
public:
    PeriodicKernel(double variance, double length_scale, double period);
    [[nodiscard]] double operator()(double t1, double t2) const override;
    [[nodiscard]] Eigen::Index num_hyperparameters() const override { return 3; }
    [[nodiscard]] Eigen::VectorXd log_hyperparameters() const override;  // [log variance, log length, log period]
    void set_log_hyperparameters(const Eigen::VectorXd& log_theta) override;
    [[nodiscard]] Eigen::VectorXd gradient(double t1, double t2) const override;
    [[nodiscard]] std::unique_ptr<IKernel> clone() const override;
    [[nodiscard]] double variance() const { return variance_; }
    [[nodiscard]] double length_scale() const { return length_; }
    [[nodiscard]] double period() const { return period_; }

private:
    double variance_;
    double length_;
    double period_;
};

/// k = k1 + k2: e.g. a smooth trend plus a seasonal component.
/// Hyperparameters are those of k1 followed by those of k2.
class SumKernel: public IKernel {
public:
    SumKernel(std::unique_ptr<IKernel> k1, std::unique_ptr<IKernel> k2);
    [[nodiscard]] double operator()(double t1, double t2) const override;
    [[nodiscard]] Eigen::Index num_hyperparameters() const override;
    [[nodiscard]] Eigen::VectorXd log_hyperparameters() const override;
    void set_log_hyperparameters(const Eigen::VectorXd& log_theta) override;
    [[nodiscard]] Eigen::VectorXd gradient(double t1, double t2) const override;
    [[nodiscard]] std::unique_ptr<IKernel> clone() const override;

private:
    std::unique_ptr<IKernel> k1_;
    std::unique_ptr<IKernel> k2_;
};

/// k = k1 * k2: e.g. a periodic pattern whose shape drifts slowly (periodic x squared exponential).
/// Hyperparameters are those of k1 followed by those of k2.
class ProductKernel: public IKernel {
public:
    ProductKernel(std::unique_ptr<IKernel> k1, std::unique_ptr<IKernel> k2);
    [[nodiscard]] double operator()(double t1, double t2) const override;
    [[nodiscard]] Eigen::Index num_hyperparameters() const override;
    [[nodiscard]] Eigen::VectorXd log_hyperparameters() const override;
    void set_log_hyperparameters(const Eigen::VectorXd& log_theta) override;
    [[nodiscard]] Eigen::VectorXd gradient(double t1, double t2) const override;
    [[nodiscard]] std::unique_ptr<IKernel> clone() const override;

private:
    std::unique_ptr<IKernel> k1_;
    std::unique_ptr<IKernel> k2_;
};

#endif // NABIA_KERNELS_H
