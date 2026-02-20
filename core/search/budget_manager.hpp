#pragma once

#include <chrono>

namespace sare {

/// Manages computational budget for search operations.
/// Tracks wall-clock time and expansion count.
class BudgetManager {
public:
    BudgetManager(double max_seconds, int max_expansions)
        : max_seconds_(max_seconds), max_expansions_(max_expansions) {}

    void start() {
        start_time_ = std::chrono::steady_clock::now();
        expansions_ = 0;
    }

    void recordExpansion() { expansions_++; }

    bool canContinue() const {
        if (expansions_ >= max_expansions_) return false;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        return elapsed < max_seconds_;
    }

    double elapsedSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    int expansions() const { return expansions_; }
    bool isTimeExhausted() const { return elapsedSeconds() >= max_seconds_; }
    bool isExpansionExhausted() const { return expansions_ >= max_expansions_; }

private:
    double max_seconds_;
    int max_expansions_;
    int expansions_ = 0;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace sare
