// Private helper: evaluates an objective on many points with a fixed pool of threads.
#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#include <Eigen/Dense>

namespace nabia::nongradopt::detail {

/// Thread pool created once per optimization run. evaluate() hands out the
/// points through an atomic counter (load balancing for uneven costs); the
/// calling thread works too. A mutex/condition-variable handshake starts each
/// batch and waits for all workers to finish it, which also makes every write
/// of the batch visible to the caller. (std::barrier would do the same, but
/// ThreadSanitizer cannot see through libc++'s implementation and reports
/// false data races.)
class ParallelEvaluator {
public:
    using Objective = std::function<double(const Eigen::VectorXd&)>;

    ParallelEvaluator(int num_threads, const Objective& objective) : objective_(objective) {
        workers_.reserve(static_cast<std::size_t>(num_threads - 1));
        for (int t = 1; t < num_threads; ++t) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ParallelEvaluator() {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        // std::jthread joins on destruction
    }

    ParallelEvaluator(const ParallelEvaluator&) = delete;
    ParallelEvaluator& operator=(const ParallelEvaluator&) = delete;

    /// values(j) = objective(points.col(j)) for every column. NaN becomes +inf.
    /// Rethrows the first exception raised by the objective, after the batch ends.
    void evaluate(const Eigen::MatrixXd& points, Eigen::VectorXd& values) {
        points_ = &points;
        values_ = &values;
        next_ = 0;
        error_ = nullptr;
        {
            const std::lock_guard lock(mutex_);
            ++batch_;
            pending_ = workers_.size();
        }
        start_cv_.notify_all();
        work();
        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this] { return pending_ == 0; });
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

    [[nodiscard]] int num_threads() const { return static_cast<int>(workers_.size()) + 1; }

private:
    void worker_loop() {
        std::size_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                start_cv_.wait(lock, [&] { return stop_ || batch_ != seen; });
                if (stop_) {
                    return;
                }
                seen = batch_;
            }
            work();
            const std::lock_guard lock(mutex_);
            if (--pending_ == 0) {
                done_cv_.notify_one();
            }
        }
    }

    void work() {
        const Eigen::Index n = points_->cols();
        for (Eigen::Index j = next_++; j < n; j = next_++) {
            try {
                const double v = objective_(points_->col(j));
                (*values_)(j) = std::isnan(v) ? std::numeric_limits<double>::infinity() : v;
            } catch (...) {
                const std::lock_guard lock(error_mutex_);
                if (!error_) {
                    error_ = std::current_exception();
                }
            }
        }
    }

    const Objective& objective_;
    const Eigen::MatrixXd* points_ = nullptr;
    Eigen::VectorXd* values_ = nullptr;
    std::atomic<Eigen::Index> next_{0};
    std::exception_ptr error_;
    std::mutex error_mutex_;
    std::mutex mutex_;                // guards batch_, pending_, stop_
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::size_t batch_ = 0;           // incremented to start a batch
    std::size_t pending_ = 0;         // workers still busy with the current batch
    bool stop_ = false;
    std::vector<std::jthread> workers_;  // last member: joined before the members above are destroyed
};

} // namespace nabia::nongradopt::detail
