// Private helper: a thread that joins when it is destroyed.
#pragma once

#include <thread>
#include <utility>

namespace nabia::detail {

/// std::thread joined by its destructor: what std::jthread does, without its
/// stop token, for standard libraries that lack std::jthread (the libc++ of
/// Xcode 15, used for the macOS wheels). Objects that the thread uses and that
/// are declared before it outlive it.
class JoiningThread {
public:
    template <typename F>
    explicit JoiningThread(F&& f) : thread_(std::forward<F>(f)) {}

    JoiningThread(JoiningThread&&) noexcept = default;
    JoiningThread& operator=(JoiningThread&&) = delete;
    JoiningThread(const JoiningThread&) = delete;
    JoiningThread& operator=(const JoiningThread&) = delete;

    ~JoiningThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

} // namespace nabia::detail
