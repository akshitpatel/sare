#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace sare {

// ─── Strategy ──────────────────────────────────────────────────
// Maps a graph structural signature to an effective transform sequence.
// Used for replay: when we see a similar problem, apply the known-good
// transform sequence directly before resorting to full search.

struct Strategy {
    std::string signature;      // graph structure hash
    std::vector<std::string> transform_sequence;
    double avg_energy_reduction = 0.0;
    int usage_count = 0;
    double success_rate = 0.0;
};

// ─── Strategy Memory ───────────────────────────────────────────
// Signature → Strategy mapping for quick replay. Decays unused
// strategies to prevent memory bloat.

class StrategyMemory {
public:
    StrategyMemory() = default;

    /// Record a successful strategy for a given graph signature.
    void record(const std::string& signature, const Strategy& strategy);

    /// Lookup the best strategy for a given signature.
    std::optional<Strategy> lookup(const std::string& signature) const;

    /// Decay all strategies by a factor (0 < factor < 1).
    /// Removes strategies with usage_count that decays to zero.
    void decay(double factor = 0.9);

    /// Number of stored strategies.
    size_t count() const { return strategies_.size(); }

    /// Remove strategies below a minimum utility threshold.
    void prune(double min_success_rate = 0.1);

    /// Clear all strategies.
    void clear() { strategies_.clear(); }

private:
    std::unordered_map<std::string, Strategy> strategies_;
};

} // namespace sare
