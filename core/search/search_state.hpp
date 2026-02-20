#pragma once

#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sare {

/// Represents a single state in the search tree.
/// Each state is a snapshot of the graph at a particular search depth.
struct SearchState {
    uint64_t id = 0;
    uint64_t snapshot_id = 0;       // Reference to SnapshotManager
    EnergyBreakdown energy;
    double score = 0.0;             // -E_total + κ * H(G)
    std::vector<std::string> transform_trace;  // Names of transforms applied
    int depth = 0;
    uint64_t parent_id = 0;         // 0 = root

    bool operator<(const SearchState& other) const {
        return score < other.score;  // higher score = better
    }
};

/// Search configuration parameters.
struct SearchConfig {
    int beam_width = 8;             // Number of states to keep per depth level
    int max_depth = 50;             // Maximum search depth
    double kappa = 0.1;             // Weight for heuristic H(G) in scoring
    double budget_seconds = 30.0;   // Maximum wall-clock time
    int max_expansions = 10000;     // Maximum state expansions
    double utility_threshold = -0.5; // Minimum transform utility to consider
};

/// Result of a search run.
struct SearchResult {
    SearchState best_state;
    Graph best_graph;
    int total_expansions = 0;
    int max_depth_reached = 0;
    double elapsed_seconds = 0.0;
    bool budget_exhausted = false;
    std::vector<SearchState> final_beam;  // Final beam of states
};

} // namespace sare
