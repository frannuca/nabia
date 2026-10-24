// Private helper: evaluates a function on many inputs with a few threads.
#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace nabia::nongradopt::detail {

/// Work-queue evaluation with promises and futures, for any job type T and
/// result type TResult. evaluate() wraps each input in a std::packaged_task,
/// keeps its std::future and enqueues the task; `num_threads` threads pop tasks
/// from the queue and run them until the queue is empty. The caller then
/// collects every result, in input order, with future::get(), which waits for
/// the task and rethrows its exception if it threw (the first failing input,
/// in order, is the one reported).
///
/// The objective is called concurrently from several threads and must be
/// thread-safe; the inputs must stay alive during evaluate() (they are read in place).
template <typename T, typename TResult>
class ParallelEvaluator {
public:
    using Objective = std::function<TResult(const T&)>;

    ParallelEvaluator(int num_threads, Objective objective)
        : num_threads_(num_threads < 1 ? 1 : num_threads), objective_(std::move(objective)) {}

    /// results[i] = objective(inputs[i]) for every input.
    [[nodiscard]] std::vector<TResult> evaluate(const std::vector<T>& inputs) const {
        std::queue<std::packaged_task<TResult()>> jobs;
        std::vector<std::future<TResult>> futures;
        futures.reserve(inputs.size());
        for (const T& input : inputs) {
            std::packaged_task<TResult()> job([this, &input] { return objective_(input); });
            futures.push_back(job.get_future());
            jobs.push(std::move(job));
        }

        std::mutex mutex;  // guards jobs
        auto consume = [&] {
            for (;;) {
                std::packaged_task<TResult()> job;
                {
                    const std::lock_guard lock(mutex);
                    if (jobs.empty()) {
                        return;
                    }
                    job = std::move(jobs.front());
                    jobs.pop();
                }
                job();  // stores the result, or the exception, in the job's promise
            }
        };

        // Declared after jobs and mutex: on any exit the consumers are joined
        // (and finish the queue) before what they use is destroyed.
        std::vector<std::jthread> consumers;
        if (num_threads_ == 1) {
            consume();
        } else {
            consumers.reserve(static_cast<std::size_t>(num_threads_));
            for (int t = 0; t < num_threads_; ++t) {
                consumers.emplace_back(consume);
            }
        }
        std::vector<TResult> results;
        results.reserve(inputs.size());
        for (std::future<TResult>& f : futures) {
            results.push_back(f.get());  // waits; rethrows the job's exception
        }
        return results;
    }

    [[nodiscard]] int num_threads() const { return num_threads_; }

private:
    int num_threads_;
    Objective objective_;
};

} // namespace nabia::nongradopt::detail
