#include "memory/strategy_memory.hpp"
#include <algorithm>

namespace sare {

void StrategyMemory::record(const std::string& signature, const Strategy& strategy) {
    auto it = strategies_.find(signature);
    if (it != strategies_.end()) {
        // Merge: update with running average
        auto& existing = it->second;
        existing.usage_count++;
        double alpha = 1.0 / existing.usage_count;
        existing.avg_energy_reduction =
            (1.0 - alpha) * existing.avg_energy_reduction +
            alpha * strategy.avg_energy_reduction;
        existing.success_rate =
            (1.0 - alpha) * existing.success_rate +
            alpha * strategy.success_rate;
        // Keep the more effective transform sequence
        if (strategy.avg_energy_reduction > existing.avg_energy_reduction) {
            existing.transform_sequence = strategy.transform_sequence;
        }
    } else {
        strategies_[signature] = strategy;
        strategies_[signature].usage_count = 1;
    }
}

std::optional<Strategy> StrategyMemory::lookup(const std::string& signature) const {
    auto it = strategies_.find(signature);
    if (it != strategies_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void StrategyMemory::decay(double factor) {
    std::vector<std::string> to_remove;
    for (auto& [sig, strat] : strategies_) {
        strat.avg_energy_reduction *= factor;
        strat.success_rate *= factor;
        // Mark for removal if decayed to negligible
        if (strat.avg_energy_reduction < 0.001 && strat.success_rate < 0.01) {
            to_remove.push_back(sig);
        }
    }
    for (const auto& sig : to_remove) {
        strategies_.erase(sig);
    }
}

void StrategyMemory::prune(double min_success_rate) {
    std::vector<std::string> to_remove;
    for (const auto& [sig, strat] : strategies_) {
        if (strat.success_rate < min_success_rate) {
            to_remove.push_back(sig);
        }
    }
    for (const auto& sig : to_remove) {
        strategies_.erase(sig);
    }
}

} // namespace sare
