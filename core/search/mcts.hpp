#pragma once

#include "search/search_state.hpp"
#include "search/budget_manager.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_registry.hpp"

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <random>

namespace sare {

// ─── MCTS Node ─────────────────────────────────────────────────
// Each node in the search tree represents a graph state.

struct MCTSNode {
    uint64_t id = 0;
    uint64_t parent_id = 0;
    std::vector<uint64_t> children;
    std::string transform_applied;  // transform that led to this state
    Graph graph;
    EnergyBreakdown energy;
    double total_reward = 0.0;
    int visit_count = 0;
    bool expanded = false;

    double ucb1(double exploration_weight = 1.414, int parent_visits = 1) const {
        if (visit_count == 0) return 1e9;  // unvisited = max priority
        double exploitation = total_reward / visit_count;
        double exploration = exploration_weight *
            std::sqrt(std::log(static_cast<double>(parent_visits)) / visit_count);
        return exploitation + exploration;
    }
};

// ─── Heuristic Function ────────────────────────────────────────
// Callback type for the learned heuristic H(G).
// In Phase 1 this is nullptr/zero. In Phase 2 the Python heuristic
// model is bridged via PyBind11.

using HeuristicFn = std::function<double(const Graph&)>;

// ─── MCTS Search ───────────────────────────────────────────────
// Monte Carlo Tree Search with:
// - UCB1 selection with heuristic prior
// - Transform-based expansion
// - Energy-evaluated rollout
// - Backpropagation of rewards

class MCTSSearch {
public:
    MCTSSearch() = default;

    /// Set the heuristic function. If null, pure energy-based scoring.
    void setHeuristic(HeuristicFn fn) { heuristic_ = fn; }

    /// Run MCTS search.
    SearchResult search(
        const Graph& initial_graph,
        const EnergyAggregator& energy,
        const TransformRegistry& transforms,
        const SearchConfig& config
    );

private:
    HeuristicFn heuristic_;
    std::vector<MCTSNode> nodes_;
    std::mt19937 rng_{42};
    uint64_t next_node_id_ = 0;

    /// Create root node.
    uint64_t createRoot(const Graph& graph, const EnergyAggregator& energy);

    /// Selection: walk down tree via UCB1.
    uint64_t select(uint64_t root_id);

    /// Expansion: expand a leaf node by applying applicable transforms.
    std::vector<uint64_t> expand(uint64_t node_id,
                                  const EnergyAggregator& energy,
                                  const TransformRegistry& transforms);

    /// Rollout: simulate forward from a node using random transform application.
    double rollout(uint64_t node_id,
                   const EnergyAggregator& energy,
                   const TransformRegistry& transforms,
                   int max_depth);

    /// Backpropagate reward up to root.
    void backpropagate(uint64_t node_id, double reward);

    /// Score a state for reward computation.
    double computeReward(const EnergyBreakdown& energy, const Graph& graph);
};

} // namespace sare
