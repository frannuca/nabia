//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_IRISKMEASURE_H
#define NABIA_IRISKMEASURE_H
#include <Eigen/Dense>

/// Risk of a portfolio x (weights or positions) whose asset returns have
/// covariance matrix Sigma.
///
/// All measures in nabia are positively homogeneous of degree one,
/// R(t x) = t R(x), so Euler's theorem gives the decomposition
///     R(x) = sum_i x_i * dR/dx_i,
/// which is what the contribution methods return:
///   - marginal contributions: the gradient dR/dx,
///   - risk contributions:     x_i * dR/dx_i (they sum to R(x)).
class IRiskMeasure {
public:
    /// Throws std::invalid_argument unless `covariance_matrix` is square,
    /// symmetric and positive semi-definite.
    explicit IRiskMeasure(Eigen::MatrixXd covariance_matrix);
    virtual ~IRiskMeasure() = default;

    [[nodiscard]] virtual double compute_risk(const Eigen::VectorXd& x) const = 0;
    [[nodiscard]] virtual Eigen::VectorXd compute_risk_contributions(const Eigen::VectorXd& x) const;
    [[nodiscard]] virtual Eigen::VectorXd compute_marginal_risk_contributions(const Eigen::VectorXd& x) const = 0;

    [[nodiscard]] Eigen::Index dim() const { return covariance_matrix_.rows(); }
    [[nodiscard]] const Eigen::MatrixXd& covariance_matrix() const { return covariance_matrix_; }

protected:
    /// sigma(x) = sqrt(x' Sigma x). Throws if x does not have dim() entries.
    [[nodiscard]] double portfolio_volatility(const Eigen::VectorXd& x) const;

    /// Gradient of sigma: Sigma x / sigma(x), or zero when sigma(x) = 0
    /// (zero is a valid subgradient there, since sigma is a seminorm).
    [[nodiscard]] Eigen::VectorXd portfolio_volatility_gradient(const Eigen::VectorXd& x) const;

    void check_dimension(const Eigen::VectorXd& v, const char* what) const;

    Eigen::MatrixXd covariance_matrix_;
};

#endif // NABIA_IRISKMEASURE_H
