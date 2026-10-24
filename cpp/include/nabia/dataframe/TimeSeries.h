//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_TIMESERIES_H
#define NABIA_TIMESERIES_H
#include "nabia/dataframe/TimeIndex.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

namespace nabia::dataframe::detail {

template <TimeIndex Index>
void require_strictly_increasing(const std::vector<Index>& index, const char* who) {
    for (std::size_t i = 1; i < index.size(); ++i) {
        if (!(index[i - 1] < index[i])) {
            throw std::invalid_argument(std::string(who) + ": index must be strictly increasing");
        }
    }
}

/// Positions [begin, end) of the index values in the closed range [from, to].
template <TimeIndex Index>
std::pair<Eigen::Index, Eigen::Index> range_positions(const std::vector<Index>& index, const Index& from, const Index& to) {
    const auto lo = std::lower_bound(index.begin(), index.end(), from);
    const auto hi = std::upper_bound(index.begin(), index.end(), to);
    const auto begin = static_cast<Eigen::Index>(lo - index.begin());
    return {begin, std::max(begin, static_cast<Eigen::Index>(hi - index.begin()))};
}

inline void require_range(Eigen::Index begin, Eigen::Index end, Eigen::Index size, const char* who) {
    if (begin < 0 || end < begin || end > size) {
        throw std::out_of_range(std::string(who) + ": row range out of bounds");
    }
}

} // namespace nabia::dataframe::detail

/// A single named column of values over a strictly increasing time index.
template <TimeIndex Index>
class TimeSeries {
public:
    TimeSeries() = default;

    /// Throws std::invalid_argument if the sizes differ or the index is not strictly increasing.
    TimeSeries(std::vector<Index> index, Eigen::VectorXd values, std::string name = "")
        : index_(std::move(index)), values_(std::move(values)), name_(std::move(name)) {
        if (static_cast<Eigen::Index>(index_.size()) != values_.size()) {
            throw std::invalid_argument("TimeSeries: index and values must have the same size");
        }
        nabia::dataframe::detail::require_strictly_increasing(index_, "TimeSeries");
    }

    [[nodiscard]] Eigen::Index size() const { return values_.size(); }
    [[nodiscard]] bool empty() const { return values_.size() == 0; }
    [[nodiscard]] const std::vector<Index>& index() const { return index_; }
    [[nodiscard]] const Eigen::VectorXd& values() const { return values_; }
    [[nodiscard]] const std::string& name() const { return name_; }

    /// Value and time at position i (no bounds check, like Eigen).
    [[nodiscard]] double operator[](Eigen::Index i) const { return values_(i); }
    [[nodiscard]] const Index& time(Eigen::Index i) const { return index_[static_cast<std::size_t>(i)]; }

    /// Position of the exact time `t`, if present (binary search).
    [[nodiscard]] std::optional<Eigen::Index> position(const Index& t) const {
        const auto it = std::lower_bound(index_.begin(), index_.end(), t);
        if (it == index_.end() || t < *it) {
            return std::nullopt;
        }
        return static_cast<Eigen::Index>(it - index_.begin());
    }

    /// Value at time `t`. Throws std::out_of_range if `t` is not in the index.
    [[nodiscard]] double at(const Index& t) const {
        const auto i = position(t);
        if (!i) {
            throw std::out_of_range("TimeSeries::at: time not in index");
        }
        return values_(*i);
    }

    /// Rows [begin, end) by position.
    [[nodiscard]] TimeSeries slice(Eigen::Index begin, Eigen::Index end) const {
        nabia::dataframe::detail::require_range(begin, end, size(), "TimeSeries::slice");
        return TimeSeries(std::vector<Index>(index_.begin() + begin, index_.begin() + end),
                          values_.segment(begin, end - begin), name_);
    }
    /// Rows with from <= time <= to.
    [[nodiscard]] TimeSeries between(const Index& from, const Index& to) const {
        const auto [b, e] = nabia::dataframe::detail::range_positions(index_, from, to);
        return slice(b, e);
    }
    [[nodiscard]] TimeSeries head(Eigen::Index n) const { return slice(0, std::min(n, size())); }
    [[nodiscard]] TimeSeries tail(Eigen::Index n) const { return slice(size() - std::min(n, size()), size()); }

private:
    std::vector<Index> index_;
    Eigen::VectorXd values_;
    std::string name_;
};

#endif // NABIA_TIMESERIES_H
