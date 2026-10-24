//
// Created by Francisco Nunez on 10.10.2026.
//

#include "nabia/forecast/IForecaster.h"

#include "nabia/stats/normal.hpp"

#include <stdexcept>

namespace {

double interval_z(double level) {
    if (!(level > 0.0 && level < 1.0)) {
        throw std::invalid_argument("Forecast: interval level must be in (0, 1)");
    }
    return nabia::stats::normal_quantile(0.5 * (1.0 + level));
}

} // namespace

Eigen::VectorXd Forecast::lower(double level) const {
    return mean - interval_z(level) * stddev();
}

Eigen::VectorXd Forecast::upper(double level) const {
    return mean + interval_z(level) * stddev();
}
