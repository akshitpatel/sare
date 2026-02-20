#pragma once

#include "graph/graph.hpp"
#include <functional>
#include <vector>

namespace sare {

/// Mapping from pattern node IDs to target node IDs.
using SubgraphMapping = std::unordered_map<uint64_t, uint64_t>;

/// Pattern constraint for a single node.
struct NodePattern {
    uint64_t pattern_id;
    std::string required_type;  // empty = any type
    std::unordered_map<std::string, std::string> required_attributes; // all must match
};

/// Pattern constraint for a single edge.
struct EdgePattern {
    uint64_t source_pattern_id;
    uint64_t target_pattern_id;
    std::string required_relationship_type;  // empty = any
};

/// Result of a Maximum Common Subgraph (MCS) search.
struct McsResult {
    SubgraphMapping mapping; // g1 node -> g2 node
    int score;               // similarity score
};

/// A subgraph pattern to search for in a target graph.
/// Defines structural and attribute constraints.
struct MatchPattern {
    std::vector<NodePattern> nodes;
    std::vector<EdgePattern> edges;
};

/// VF2-inspired subgraph isomorphism matcher.
/// Finds all mappings from pattern → target graph that satisfy constraints.
class SubgraphMatcher {
public:
    /// Find all subgraph mappings of pattern in target.
    static std::vector<SubgraphMapping> match(
        const MatchPattern& pattern,
        const Graph& target
    );

    /// Find the Maximum Common Subgraph between g1 and g2.
    /// Limits search depth to keep worst-case complexity manageable.
    static McsResult find_mcs(
        const Graph& g1,
        const Graph& g2,
        int max_depth = 8
    );

private:
    struct MatchState {
        const MatchPattern* pattern;
        const Graph* target;
        SubgraphMapping mapping;
        std::unordered_set<uint64_t> used_target_nodes;
        std::vector<SubgraphMapping> results;
    };

    struct McsState {
        const Graph* g1;
        const Graph* g2;
        int max_depth;
        std::vector<uint64_t> g1_nodes; // nodes we iterate over
        
        SubgraphMapping current_mapping;
        std::unordered_set<uint64_t> used_g2_nodes;
        int current_score = 0;
        
        McsResult best_result;
    };

    static void backtrack(MatchState& state, size_t pattern_node_idx);
    static bool isCompatible(const NodePattern& pnode, const Node& tnode);
    static bool edgesConsistent(const MatchState& state, size_t pattern_node_idx);
    
    static void mcs_backtrack(McsState& state, size_t g1_idx, int depth);
    static int score_node_match(const Node& n1, const Node& n2);
    static int score_edge_match(const McsState& state, uint64_t g1_node_id, uint64_t g2_node_id);
};

} // namespace sare
