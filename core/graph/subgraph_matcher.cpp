#include "graph/subgraph_matcher.hpp"

namespace sare {

std::vector<SubgraphMapping> SubgraphMatcher::match(
    const MatchPattern& pattern,
    const Graph& target
) {
    MatchState state;
    state.pattern = &pattern;
    state.target = &target;

    if (pattern.nodes.empty()) {
        return {};
    }

    backtrack(state, 0);
    return state.results;
}

void SubgraphMatcher::backtrack(MatchState& state, size_t pattern_node_idx) {
    if (pattern_node_idx >= state.pattern->nodes.size()) {
        // All pattern nodes mapped — verify all edge constraints
        bool edges_ok = true;
        for (const auto& ep : state.pattern->edges) {
            uint64_t src_target = state.mapping.at(ep.source_pattern_id);
            uint64_t tgt_target = state.mapping.at(ep.target_pattern_id);

            // Check if an edge of the required type exists
            bool found = false;
            auto outgoing = state.target->getOutgoing(src_target);
            for (uint64_t eid : outgoing) {
                const Edge* e = state.target->getEdge(eid);
                if (e && e->target == tgt_target) {
                    if (ep.required_relationship_type.empty() ||
                        e->relationship_type == ep.required_relationship_type) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                edges_ok = false;
                break;
            }
        }
        if (edges_ok) {
            state.results.push_back(state.mapping);
        }
        return;
    }

    const NodePattern& pnode = state.pattern->nodes[pattern_node_idx];

    // Try mapping pattern_node to each target node
    auto target_ids = state.target->getNodeIds();
    for (uint64_t tid : target_ids) {
        if (state.used_target_nodes.count(tid)) continue;

        const Node* tnode = state.target->getNode(tid);
        if (!tnode) continue;

        if (!isCompatible(pnode, *tnode)) continue;

        // Tentatively map
        state.mapping[pnode.pattern_id] = tid;
        state.used_target_nodes.insert(tid);

        // Check edge consistency for already-mapped nodes
        if (edgesConsistent(state, pattern_node_idx)) {
            backtrack(state, pattern_node_idx + 1);
        }

        // Undo
        state.mapping.erase(pnode.pattern_id);
        state.used_target_nodes.erase(tid);
    }
}

bool SubgraphMatcher::isCompatible(const NodePattern& pnode, const Node& tnode) {
    // Type check
    if (!pnode.required_type.empty() && pnode.required_type != tnode.type) {
        return false;
    }
    // Attribute check
    for (const auto& [key, val] : pnode.required_attributes) {
        auto it = tnode.attributes.find(key);
        if (it == tnode.attributes.end() || it->second != val) {
            return false;
        }
    }
    return true;
}

bool SubgraphMatcher::edgesConsistent(const MatchState& state, size_t pattern_node_idx) {
    // Check edge constraints involving the current pattern node and already-mapped nodes
    const NodePattern& current = state.pattern->nodes[pattern_node_idx];

    for (const auto& ep : state.pattern->edges) {
        bool involves_current = (ep.source_pattern_id == current.pattern_id ||
                                  ep.target_pattern_id == current.pattern_id);
        if (!involves_current) continue;

        // Both endpoints must be mapped for this check
        bool src_mapped = state.mapping.count(ep.source_pattern_id) > 0;
        bool tgt_mapped = state.mapping.count(ep.target_pattern_id) > 0;
        if (!src_mapped || !tgt_mapped) continue;

        uint64_t src_target = state.mapping.at(ep.source_pattern_id);
        uint64_t tgt_target = state.mapping.at(ep.target_pattern_id);

        bool found = false;
        auto outgoing = state.target->getOutgoing(src_target);
        for (uint64_t eid : outgoing) {
            const Edge* e = state.target->getEdge(eid);
            if (e && e->target == tgt_target) {
                if (ep.required_relationship_type.empty() ||
                    e->relationship_type == ep.required_relationship_type) {
                    found = true;
                    break;
                }
            }
        }
        if (!found) return false;
    }
    return true;
}

McsResult SubgraphMatcher::find_mcs(
    const Graph& g1,
    const Graph& g2,
    int max_depth
) {
    McsState state;
    state.g1 = &g1;
    state.g2 = &g2;
    state.max_depth = max_depth;
    
    for (uint64_t nid : g1.getNodeIds()) {
        state.g1_nodes.push_back(nid);
    }
    
    state.best_result.score = -1;
    
    mcs_backtrack(state, 0, 0);
    return state.best_result;
}

void SubgraphMatcher::mcs_backtrack(McsState& state, size_t g1_idx, int depth) {
    if (g1_idx >= state.g1_nodes.size() || depth >= state.max_depth) {
        if (state.current_score > state.best_result.score) {
            state.best_result.score = state.current_score;
            state.best_result.mapping = state.current_mapping;
        }
        return;
    }
    
    uint64_t n1_id = state.g1_nodes[g1_idx];
    const Node* n1 = state.g1->getNode(n1_id);
    if (!n1) {
        mcs_backtrack(state, g1_idx + 1, depth); 
        return;
    }

    // Branch 1: Leave this node unmapped
    mcs_backtrack(state, g1_idx + 1, depth);

    // Branch 2: Try mapping n1 to each available node in g2
    for (uint64_t n2_id : state.g2->getNodeIds()) {
        if (state.used_g2_nodes.count(n2_id)) continue;
        
        const Node* n2 = state.g2->getNode(n2_id);
        if (!n2) continue;
        
        int n_score = score_node_match(*n1, *n2);
        if (n_score <= 0) continue; 
        
        state.current_mapping[n1_id] = n2_id;
        state.used_g2_nodes.insert(n2_id);
        
        int e_score = score_edge_match(state, n1_id, n2_id);
        int added_score = n_score + e_score;
        state.current_score += added_score;
        
        mcs_backtrack(state, g1_idx + 1, depth + 1);
        
        // Backtrack
        state.current_score -= added_score;
        state.used_g2_nodes.erase(n2_id);
        state.current_mapping.erase(n1_id);
    }
}

int SubgraphMatcher::score_node_match(const Node& n1, const Node& n2) {
    int score = 0;
    if (n1.type == n2.type) {
        score += 10;
        
        if (n1.type == "operator") {
            auto sym1 = n1.getAttribute("operator_symbol");
            auto sym2 = n2.getAttribute("operator_symbol");
            if (!sym1.empty() && sym1 == sym2) score += 5;
        } else if (n1.type == "constant") {
            auto v1 = n1.getAttribute("value");
            auto v2 = n2.getAttribute("value");
            if (!v1.empty() && v1 == v2) score += 5;
        }
    }
    return score;
}

int SubgraphMatcher::score_edge_match(const McsState& state, uint64_t g1_node_id, uint64_t g2_node_id) {
    int score = 0;
    
    auto check_edges = [&](const std::vector<uint64_t>& g1_edges, bool outgoing) {
        for (uint64_t e1_id : g1_edges) {
            const Edge* e1 = state.g1->getEdge(e1_id);
            if (!e1) continue;
            
            uint64_t partner1_id = outgoing ? e1->target : e1->source;
            auto it = state.current_mapping.find(partner1_id);
            
            if (it != state.current_mapping.end()) {
                uint64_t partner2_id = it->second;
                
                std::vector<uint64_t> g2_edges = outgoing ? 
                    state.g2->getOutgoing(g2_node_id) : state.g2->getIncoming(g2_node_id);
                    
                for (uint64_t e2_id : g2_edges) {
                    const Edge* e2 = state.g2->getEdge(e2_id);
                    if (!e2) continue;
                    
                    uint64_t p2_actual = outgoing ? e2->target : e2->source;
                    if (p2_actual == partner2_id) {
                        score += 5; 
                        if (e1->relationship_type == e2->relationship_type) {
                            score += 5; 
                        }
                        break;
                    }
                }
            }
        }
    };
    
    check_edges(state.g1->getOutgoing(g1_node_id), true);
    check_edges(state.g1->getIncoming(g1_node_id), false);
    
    return score;
}

} // namespace sare
