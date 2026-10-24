// Private helper: run independent tasks on a few threads.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace nabia::detail {

/// Number of threads to use: `requested`, or the hardware concurrency when it
/// is 0; never more than `tasks` and never less than 1.
inline int resolve_threads(int requested, std::int64_t tasks) {
    const int hardware = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int wanted = requested == 0 ? hardware : requested;
    return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(wanted, tasks)));
}

/// Calls task(i) for every i in [0, count) on `threads` threads (the caller
/// included). Tasks are handed out through an atomic counter, so uneven costs
/// balance out. If tasks throw, the remaining tasks still run and the exception
/// of the lowest failing index is rethrown, which makes errors deterministic.
template <typename Task>
void parallel_for(std::int64_t count, int threads, const Task& task) {
    std::atomic<std::int64_t> next{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    std::int64_t error_index = count;

    auto work = [&] {
        for (std::int64_t i = next++; i < count; i = next++) {
            try {
                task(i);
            } catch (...) {
                const std::lock_guard lock(error_mutex);
                if (i < error_index) {
                    error_index = i;
                    error = std::current_exception();
                }
            }
        }
    };
    {
        std::vector<std::jthread> workers;
        workers.reserve(static_cast<std::size_t>(std::max(0, threads - 1)));
        for (int t = 1; t < threads; ++t) {
            workers.emplace_back(work);
        }
        work();
    }  // joining the workers makes all their writes visible here
    if (error) {
        std::rethrow_exception(error);
    }
}

} // namespace nabia::detail
