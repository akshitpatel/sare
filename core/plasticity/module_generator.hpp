#pragma once

#include "transforms/transform_base.hpp"
#include "memory/episodic_store.hpp"
#include "graph/graph.hpp"
#include <memory>
#include <vector>

namespace sare {

// ─── Module Generator ─────────────────────────────────────────
// Proposes new transform templates when persistent failure
// patterns are detected. Part of Structural Plasticity (Phase 4).
//
// If persistent failure pattern detected:
// 1. Analyze failure episodes for common structural features
// 2. Propose a candidate transform template
// 3. Hand off to SandboxRunner for evaluation

class ModuleGenerator {
public:
    ModuleGenerator() = default;

    /// Analyze failure episodes and propose new transform candidates.
    /// Returns candidate transforms for sandbox evaluation.
    std::vector<std::unique_ptr<Transform>> generate(
        const std::vector<SolveEpisode>& failures,
        size_t max_candidates = 3);

    /// Check if failure pattern is persistent (occurs frequently enough).
    bool isPersistentFailure(const std::vector<SolveEpisode>& episodes,
                              double failure_rate_threshold = 0.5) const;

    /// Extract common structural features from failed episodes.
    std::vector<std::string> extractFailureFeatures(
        const std::vector<SolveEpisode>& failures) const;
};

} // namespace sare
