//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/risk/measures/Volatility.h"

double Volatility::compute_risk(const Eigen::VectorXd& x) const {
    return portfolio_volatility(x);
}

Eigen::VectorXd Volatility::compute_marginal_risk_contributions(const Eigen::VectorXd& x) const {
    return portfolio_volatility_gradient(x);
}
