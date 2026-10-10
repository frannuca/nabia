//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_TIMEINDEX_H
#define NABIA_TIMEINDEX_H
#include <chrono>
#include <concepts>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

/// Requirements on the time index of a TimeSeries or DataFrame: values must be
/// copyable and totally ordered (the index is kept strictly increasing).
/// Dates, timestamps and plain integers all qualify.
template <typename T>
concept TimeIndex = std::copyable<T> && std::totally_ordered<T>;

/// Common index types.
using Date = std::chrono::sys_days;                                    // calendar day
using Timestamp = std::chrono::sys_time<std::chrono::nanoseconds>;     // point in time (UTC)
using Step = std::int64_t;                                             // integer position / bar number

/// `count` consecutive days starting at `first` (calendar days, weekends included).
[[nodiscard]] inline std::vector<Date> daily_dates(Date first, Eigen::Index count) {
    std::vector<Date> out;
    out.reserve(static_cast<std::size_t>(count));
    for (Eigen::Index i = 0; i < count; ++i) {
        out.push_back(first + std::chrono::days(i));
    }
    return out;
}

/// The integers first, first + 1, ..., first + count - 1.
[[nodiscard]] inline std::vector<Step> integer_index(Eigen::Index count, Step first = 0) {
    std::vector<Step> out;
    out.reserve(static_cast<std::size_t>(count));
    for (Eigen::Index i = 0; i < count; ++i) {
        out.push_back(first + i);
    }
    return out;
}

#endif // NABIA_TIMEINDEX_H
