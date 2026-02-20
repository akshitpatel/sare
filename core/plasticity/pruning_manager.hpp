#pragma once

#include "transforms/transform_registry.hpp"
#include <string>
#include <vector>

namespace sare {

// ─── Pruning Manager ──────────────────────────────────────────
// Removes underperforming modules from the transform registry.
// Prune condition: U_k < 0
//
// This is the complementary operation to module generation:
// - ModuleGenerator: adds new modules
// - PruningManager: removes weak modules
//
// Together they implement controlled evolution of the transform set.

class PruningManager {
public:
    PruningManager(double utility_threshold = 0.0)
        : utility_threshold_(utility_threshold) {}

    /// Identify transforms with utility below threshold.
    /// Returns list of transform names to prune.
    std::vector<std::string> identifyCandidates(
        const TransformRegistry& registry) const;

    /// Prune the identified transforms from the registry.
    /// Returns number of transforms actually removed.
    int prune(TransformRegistry& registry,
              const std::vector<std::string>& names);

    /// Combined: identify and prune in one step.
    int pruneUnderperformers(TransformRegistry& registry);

    /// Set utility threshold.
    void setThreshold(double threshold) { utility_threshold_ = threshold; }
    double threshold() const { return utility_threshold_; }

private:
    double utility_threshold_;
};

} // namespace sare
