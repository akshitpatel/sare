#pragma once

#include "transforms/transform_base.hpp"
#include "transforms/transform_registry.hpp"
#include "energy/energy.hpp"
#include "graph/graph.hpp"
#include "search/beam_search.hpp"
#include "search/search_state.hpp"
#include <vector>

namespace sare {

// ─── Sandbox Result ───────────────────────────────────────────
// Result of evaluating a candidate module in an isolated environment.

struct SandboxResult {
    std::string candidate_name;
    double performance_delta = 0.0;  // Performance_with - Performance_without
    double avg_energy_reduction = 0.0;
    int problems_tested = 0;
    int problems_improved = 0;
    bool promoted = false;  // performance_delta > ε
};

// ─── Sandbox Runner ───────────────────────────────────────────
// Evaluates candidate modules in isolated environment.
// The sandbox:
// 1. Runs baseline search WITHOUT the candidate
// 2. Runs search WITH the candidate
// 3. Compares performance
// 4. Promotes only if ΔPerformance > ε

class SandboxRunner {
public:
    SandboxRunner(double promotion_threshold = 0.05)
        : promotion_threshold_(promotion_threshold) {}

    /// Evaluate a candidate transform against test problems.
    SandboxResult evaluate(
        Transform* candidate,
        const std::vector<Graph>& test_problems,
        const EnergyAggregator& energy,
        const TransformRegistry& baseline_registry,
        const SearchConfig& search_config);

private:
    double promotion_threshold_;  // ε

    /// Run search on a single problem and return best energy.
    double solveOne(const Graph& problem,
                    const EnergyAggregator& energy,
                    const TransformRegistry& registry,
                    const SearchConfig& config);
};

} // namespace sare
