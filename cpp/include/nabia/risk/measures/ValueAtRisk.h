//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_VALUEATRISK_H
#define NABIA_VALUEATRISK_H
#include "nabia/risk/IRiskMeasure.h"

/// Parametric (Gaussian, delta-normal) Value at Risk.
///
/// With asset returns r ~ N(mu, Sigma) and portfolio loss L = -r'x, VaR at
/// confidence level alpha is the alpha-quantile of L:
///     VaR_alpha(x) = -mu'x + z_alpha * sqrt(x' Sigma x),   z_alpha = Phi^-1(alpha).
/// It is reported as a positive number when the portfolio is at risk of a loss.
/// Mean and covariance must refer to the same horizon (e.g. daily returns give a
/// one-day VaR).
class ValueAtRisk: public IRiskMeasure {
public:
    /// `confidence_level` in (0, 1), e.g. 0.99. `expected_returns` empty means
    /// mu = 0; otherwise it needs one entry per asset.
    ValueAtRisk(Eigen::MatrixXd covariance_matrix, double confidence_level,
                Eigen::VectorXd expected_returns = Eigen::VectorXd());

    [[nodiscard]] double compute_risk(const Eigen::VectorXd& x) const override;
    /// -mu + z_alpha * Sigma x / sigma(x)
    [[nodiscard]] Eigen::VectorXd compute_marginal_risk_contributions(const Eigen::VectorXd& x) const override;

    [[nodiscard]] double confidence_level() const { return confidence_level_; }

private:
    double confidence_level_;
    double z_;                         // Phi^-1(alpha)
    Eigen::VectorXd expected_returns_; // mu, zero if not given
};

#endif // NABIA_VALUEATRISK_H
