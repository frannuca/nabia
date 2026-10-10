//
// Created by Francisco Nunez on 10.10.2026.
//

#ifndef NABIA_DATAFRAME_H
#define NABIA_DATAFRAME_H
#include "nabia/dataframe/TimeIndex.h"
#include "nabia/dataframe/TimeSeries.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

/// A table of named columns over a strictly increasing time index: one row per
/// time, one column per variable (e.g. one per asset). The values are a dense
/// Eigen::MatrixXd (rows = time), so numerical code can work on values()
/// directly while the index keeps track of time.
template <TimeIndex Index>
class DataFrame {
public:
    DataFrame() = default;

    /// Throws std::invalid_argument if the shapes disagree, the index is not
    /// strictly increasing or column names are duplicated.
    DataFrame(std::vector<Index> index, std::vector<std::string> columns, Eigen::MatrixXd values)
        : index_(std::move(index)), columns_(std::move(columns)), values_(std::move(values)) {
        if (static_cast<Eigen::Index>(index_.size()) != values_.rows() ||
            static_cast<Eigen::Index>(columns_.size()) != values_.cols()) {
            throw std::invalid_argument("DataFrame: index and columns must match the shape of values");
        }
        nabia::dataframe::detail::require_strictly_increasing(index_, "DataFrame");
        std::vector<std::string> sorted = columns_;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("DataFrame: column names must be unique");
        }
    }

    [[nodiscard]] Eigen::Index rows() const { return values_.rows(); }
    [[nodiscard]] Eigen::Index cols() const { return values_.cols(); }
    [[nodiscard]] bool empty() const { return values_.size() == 0; }
    [[nodiscard]] const std::vector<Index>& index() const { return index_; }
    [[nodiscard]] const std::vector<std::string>& columns() const { return columns_; }
    [[nodiscard]] const Eigen::MatrixXd& values() const { return values_; }
    [[nodiscard]] const Index& time(Eigen::Index row) const { return index_[static_cast<std::size_t>(row)]; }

    /// Column position of `name`. Throws std::out_of_range if absent.
    [[nodiscard]] Eigen::Index column_position(const std::string& name) const {
        const auto it = std::find(columns_.begin(), columns_.end(), name);
        if (it == columns_.end()) {
            throw std::out_of_range("DataFrame: no column named '" + name + "'");
        }
        return static_cast<Eigen::Index>(it - columns_.begin());
    }

    [[nodiscard]] TimeSeries<Index> column(Eigen::Index j) const {
        return TimeSeries<Index>(index_, values_.col(j), columns_[static_cast<std::size_t>(j)]);
    }
    [[nodiscard]] TimeSeries<Index> column(const std::string& name) const { return column(column_position(name)); }

    /// Row position of the exact time `t`, if present.
    [[nodiscard]] std::optional<Eigen::Index> position(const Index& t) const {
        const auto it = std::lower_bound(index_.begin(), index_.end(), t);
        if (it == index_.end() || t < *it) {
            return std::nullopt;
        }
        return static_cast<Eigen::Index>(it - index_.begin());
    }
    /// Values at time `t` (one per column). Throws std::out_of_range if absent.
    [[nodiscard]] Eigen::RowVectorXd row_at(const Index& t) const {
        const auto i = position(t);
        if (!i) {
            throw std::out_of_range("DataFrame::row_at: time not in index");
        }
        return values_.row(*i);
    }

    /// Rows [begin, end) by position.
    [[nodiscard]] DataFrame slice(Eigen::Index begin, Eigen::Index end) const {
        nabia::dataframe::detail::require_range(begin, end, rows(), "DataFrame::slice");
        return DataFrame(std::vector<Index>(index_.begin() + begin, index_.begin() + end), columns_,
                         values_.middleRows(begin, end - begin));
    }
    /// Rows with from <= time <= to.
    [[nodiscard]] DataFrame between(const Index& from, const Index& to) const {
        const auto [b, e] = nabia::dataframe::detail::range_positions(index_, from, to);
        return slice(b, e);
    }
    [[nodiscard]] DataFrame head(Eigen::Index n) const { return slice(0, std::min(n, rows())); }
    [[nodiscard]] DataFrame tail(Eigen::Index n) const { return slice(rows() - std::min(n, rows()), rows()); }

    /// The named columns, in the given order.
    [[nodiscard]] DataFrame select(const std::vector<std::string>& names) const {
        Eigen::MatrixXd out(rows(), static_cast<Eigen::Index>(names.size()));
        for (std::size_t k = 0; k < names.size(); ++k) {
            out.col(static_cast<Eigen::Index>(k)) = values_.col(column_position(names[k]));
        }
        return DataFrame(index_, names, std::move(out));
    }

    /// Rows where every column has a finite value. Rows with a NaN or an
    /// infinity in any column are dropped, which aligns series that do not all
    /// have data at the same times (e.g. markets with different holidays).
    [[nodiscard]] DataFrame drop_missing() const {
        std::vector<Index> index;
        std::vector<Eigen::Index> keep;
        for (Eigen::Index i = 0; i < rows(); ++i) {
            if (values_.row(i).allFinite()) {
                keep.push_back(i);
                index.push_back(index_[static_cast<std::size_t>(i)]);
            }
        }
        Eigen::MatrixXd out(static_cast<Eigen::Index>(keep.size()), cols());
        for (std::size_t k = 0; k < keep.size(); ++k) {
            out.row(static_cast<Eigen::Index>(k)) = values_.row(keep[k]);
        }
        return DataFrame(std::move(index), columns_, std::move(out));
    }

    /// Simple returns x[t] / x[t-1] - 1 of price-like columns. The first row is
    /// dropped, so row t of the result is the return earned over (t-1, t].
    [[nodiscard]] DataFrame pct_change() const {
        if (rows() < 2) {
            throw std::invalid_argument("DataFrame::pct_change: need at least 2 rows");
        }
        const Eigen::MatrixXd r = values_.bottomRows(rows() - 1).cwiseQuotient(values_.topRows(rows() - 1)).array() - 1.0;
        return DataFrame(std::vector<Index>(index_.begin() + 1, index_.end()), columns_, r);
    }
    /// Log returns log(x[t] / x[t-1]); first row dropped as in pct_change().
    [[nodiscard]] DataFrame log_returns() const {
        if (rows() < 2) {
            throw std::invalid_argument("DataFrame::log_returns: need at least 2 rows");
        }
        const Eigen::MatrixXd r = values_.bottomRows(rows() - 1).cwiseQuotient(values_.topRows(rows() - 1)).array().log();
        return DataFrame(std::vector<Index>(index_.begin() + 1, index_.end()), columns_, r);
    }

private:
    std::vector<Index> index_;
    std::vector<std::string> columns_;
    Eigen::MatrixXd values_;
};

#endif // NABIA_DATAFRAME_H
