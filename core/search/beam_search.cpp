#include "search/beam_search.hpp"
#include <algorithm>
#include <limits>

namespace sare {

SearchResult BeamSearch::search(
    const Graph& initial_graph,
    const EnergyAggregator& energy_agg,
    const TransformRegistry& transforms,
    const SearchConfig& config
) {
    BudgetManager budget(config.budget_seconds, config.max_expansions);
    budget.start();

    SearchResult result;
    uint64_t next_state_id = 1;

    // Initialize beam with the starting state
    SearchState initial;
    initial.id = next_state_id++;
    initial.energy = energy_agg.computeTotal(initial_graph);
    initial.score = scoreState(initial.energy, config.kappa);
    initial.depth = 0;

    std::vector<std::pair<SearchState, Graph>> beam;
    beam.push_back({initial, initial_graph.clone()});

    SearchState best = initial;
    Graph best_graph = initial_graph.clone();

    for (int depth = 0; depth < config.max_depth; depth++) {
        if (!budget.canContinue()) {
            result.budget_exhausted = true;
            break;
        }

        std::vector<std::pair<SearchState, Graph>> candidates;

        for (auto& [state, graph] : beam) {
            // Get applicable transforms
            auto applicable = transforms.getApplicable(graph, config.utility_threshold);

            for (Transform* transform : applicable) {
                if (!budget.canContinue()) break;

                budget.recordExpansion();

                // Apply transform to a copy
                GraphDelta delta = transform->apply(graph);
                if (delta.empty()) continue;

                Graph new_graph = graph.clone();
                new_graph.applyDelta(delta);

                // Score the new state
                EnergyBreakdown new_energy = energy_agg.computeTotal(new_graph);

                SearchState new_state;
                new_state.id = next_state_id++;
                new_state.energy = new_energy;
                new_state.score = scoreState(new_energy, config.kappa);
                new_state.depth = depth + 1;
                new_state.parent_id = state.id;
                new_state.transform_trace = state.transform_trace;
                new_state.transform_trace.push_back(transform->name());

                candidates.push_back({new_state, std::move(new_graph)});

                // Update best
                if (new_state.score > best.score) {
                    best = new_state;
                    best_graph = candidates.back().second.clone();
                }
            }
        }

        if (candidates.empty()) break;  // No more transforms applicable

        // Sort by score (descending) and prune to beam width
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
                return a.first.score > b.first.score;
            });

        beam.clear();
        int width = std::min(config.beam_width, static_cast<int>(candidates.size()));
        for (int i = 0; i < width; i++) {
            beam.push_back(std::move(candidates[i]));
        }

        result.max_depth_reached = depth + 1;

        // Early termination: if energy is near zero
        if (best.energy.total() < 1e-6) break;
    }

    result.best_state = best;
    result.best_graph = best_graph.clone();
    result.total_expansions = budget.expansions();
    result.elapsed_seconds = budget.elapsedSeconds();

    for (const auto& [state, _] : beam) {
        result.final_beam.push_back(state);
    }

    return result;
}

double BeamSearch::scoreState(const EnergyBreakdown& e, double /*kappa*/) const {
    // Phase 1: no heuristic model, so H(G) = 0
    // Score = -E_total (lower energy = higher score)
    return -e.total();
}

} // namespace sare
