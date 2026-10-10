// Private helpers shared by the derivative-free optimizers (not installed).
#pragma once

#include "nabia/optimization/convex/Constraints.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <Eigen/Dense>

namespace nabia::nongradopt::detail {

/// Box bounds as dense vectors (+-infinity where unbounded).
struct Box {
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;

    [[nodiscard]] Eigen::VectorXd project(const Eigen::VectorXd& x) const { return x.cwiseMax(lower).cwiseMin(upper); }
    [[nodiscard]] bool finite() const { return lower.allFinite() && upper.allFinite(); }
};

inline Box make_box(const std::optional<BoxConstraints>& bounds, Eigen::Index n, const char* who) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    Box box{Eigen::VectorXd::Constant(n, -inf), Eigen::VectorXd::Constant(n, inf)};
    if (!bounds) {
        return box;
    }
    const auto size = static_cast<std::size_t>(n);
    const auto& lo = bounds->lower_bounds;
    const auto& up = bounds->upper_bounds;
    if ((!lo.empty() && lo.size() != size) || (!up.empty() && up.size() != size)) {
        throw std::invalid_argument(std::string(who) + ": bounds must be empty or have `dim` entries");
    }
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto j = static_cast<std::size_t>(i);
        if (!lo.empty()) {
            box.lower(i) = lo[j];
        }
        if (!up.empty()) {
            box.upper(i) = up[j];
        }
        if (std::isnan(box.lower(i)) || std::isnan(box.upper(i)) || box.lower(i) > box.upper(i)) {
            throw std::invalid_argument(std::string(who) + ": bounds need lower <= upper");
        }
    }
    return box;
}

} // namespace nabia::nongradopt::detail
