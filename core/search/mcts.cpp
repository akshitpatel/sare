#include "search/mcts.hpp"
#include <algorithm>
#include <limits>

namespace sare {

uint64_t MCTSSearch::createRoot(const Graph& graph, const EnergyAggregator& energy) {
    nodes_.clear();
    next_node_id_ = 0;

    MCTSNode root;
    root.id = next_node_id_++;
    root.graph = graph.clone();
    root.energy = energy.computeTotal(graph);
    root.visit_count = 1;
    nodes_.push_back(std::move(root));
    return 0;
}

uint64_t MCTSSearch::select(uint64_t root_id) {
    uint64_t current = root_id;

    while (nodes_[current].expanded && !nodes_[current].children.empty()) {
        // Pick child with highest UCB1
        double best_ucb = -std::numeric_limits<double>::infinity();
        uint64_t best_child = nodes_[current].children[0];

        for (uint64_t child_id : nodes_[current].children) {
            double ucb = nodes_[child_id].ucb1(1.414, nodes_[current].visit_count);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child_id;
            }
        }
        current = best_child;
    }

    return current;
}

std::vector<uint64_t> MCTSSearch::expand(uint64_t node_id,
                                          const EnergyAggregator& energy,
                                          const TransformRegistry& transforms) {
    MCTSNode& node = nodes_[node_id];
    if (node.expanded) return node.children;

    node.expanded = true;

    auto applicable = transforms.getApplicable(node.graph);
    for (Transform* transform : applicable) {
        if (!transform->match(node.graph)) continue;

        GraphDelta delta = transform->apply(node.graph);
        if (delta.empty()) continue;

        Graph new_graph = node.graph.clone();
        new_graph.applyDelta(delta);

        MCTSNode child;
        child.id = next_node_id_++;
        child.parent_id = node_id;
        child.transform_applied = transform->name();
        child.graph = std::move(new_graph);
        child.energy = energy.computeTotal(child.graph);

        node.children.push_back(child.id);
        nodes_.push_back(std::move(child));
    }

    return node.children;
}

double MCTSSearch::rollout(uint64_t node_id,
                            const EnergyAggregator& energy,
                            const TransformRegistry& transforms,
                            int max_depth) {
    Graph current_graph = nodes_[node_id].graph.clone();
    double best_energy = nodes_[node_id].energy.total();

    for (int d = 0; d < max_depth; d++) {
        auto applicable = transforms.getApplicable(current_graph);
        if (applicable.empty()) break;

        // Randomly pick a transform
        std::uniform_int_distribution<size_t> dist(0, applicable.size() - 1);
        Transform* chosen = applicable[dist(rng_)];

        if (!chosen->match(current_graph)) continue;

        GraphDelta delta = chosen->apply(current_graph);
        if (delta.empty()) continue;

        current_graph.applyDelta(delta);
        double e = energy.computeTotal(current_graph).total();
        best_energy = std::min(best_energy, e);

        if (e < 0.001) break;  // near-zero: solved
    }

    return computeReward(energy.computeTotal(current_graph), current_graph);
}

void MCTSSearch::backpropagate(uint64_t node_id, double reward) {
    uint64_t current = node_id;
    while (true) {
        nodes_[current].visit_count++;
        nodes_[current].total_reward += reward;
        if (current == 0) break;  // root
        current = nodes_[current].parent_id;
    }
}

double MCTSSearch::computeReward(const EnergyBreakdown& energy, const Graph& graph) {
    // Reward = negative energy (higher reward for lower energy)
    double reward = -energy.total();

    // Add heuristic bonus if available
    if (heuristic_) {
        reward += heuristic_(graph);
    }

    return reward;
}

SearchResult MCTSSearch::search(
    const Graph& initial_graph,
    const EnergyAggregator& energy,
    const TransformRegistry& transforms,
    const SearchConfig& config) {

    uint64_t root_id = createRoot(initial_graph, energy);

    BudgetManager budget(config.budget_seconds, config.max_expansions);
    budget.start();

    SearchResult result;
    result.best_state.energy = nodes_[root_id].energy;
    result.best_state.score = computeReward(result.best_state.energy, initial_graph);

    uint64_t best_node_id = root_id;

    while (budget.canContinue()) {
        // 1. Selection
        uint64_t selected = select(root_id);

        // 2. Expansion
        auto children = expand(selected, energy, transforms);
        budget.recordExpansion();

        if (children.empty()) continue;

        // Pick a random unexplored child for rollout
        uint64_t rollout_node = children[0];
        for (uint64_t cid : children) {
            if (nodes_[cid].visit_count == 0) {
                rollout_node = cid;
                break;
            }
        }

        // 3. Rollout
        int rollout_depth = std::max(1, config.max_depth / 5);
        double reward = rollout(rollout_node, energy, transforms, rollout_depth);

        // 4. Backpropagate
        backpropagate(rollout_node, reward);

        // Track best
        double node_energy = nodes_[rollout_node].energy.total();
        if (node_energy < result.best_state.energy.total()) {
            result.best_state.energy = nodes_[rollout_node].energy;
            result.best_state.score = computeReward(result.best_state.energy,
                                                     nodes_[rollout_node].graph);
            best_node_id = rollout_node;
        }
    }

    // Build transform trace for best node
    result.best_state.transform_trace.clear();
    uint64_t trace_node = best_node_id;
    while (trace_node != root_id) {
        if (!nodes_[trace_node].transform_applied.empty()) {
            result.best_state.transform_trace.push_back(
                nodes_[trace_node].transform_applied);
        }
        trace_node = nodes_[trace_node].parent_id;
    }
    std::reverse(result.best_state.transform_trace.begin(),
                 result.best_state.transform_trace.end());

    result.total_expansions = budget.expansions();
    result.elapsed_seconds = budget.elapsedSeconds();
    result.budget_exhausted = !budget.canContinue();
    result.max_depth_reached = static_cast<int>(result.best_state.transform_trace.size());
    result.best_graph = nodes_[best_node_id].graph.clone();

    return result;
}

} // namespace sare
