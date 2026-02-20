#pragma once

#include "search/search_state.hpp"
#include "search/budget_manager.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_registry.hpp"

namespace sare {

/// Beam Search: deterministic energy-minimizing search.
/// At each depth, expands all beam states via applicable transforms,
/// scores resulting states, and prunes to beam_width.
class BeamSearch {
public:
    /// Run beam search on the given graph.
    /// Returns the best solution found within budget.
    SearchResult search(
        const Graph& initial_graph,
        const EnergyAggregator& energy,
        const TransformRegistry& transforms,
        const SearchConfig& config
    );

private:
    /// Score a state: -E_total + κ * H(G)
    /// In Phase 1, H(G) = 0 (no heuristic model yet).
    double scoreState(const EnergyBreakdown& energy, double kappa) const;
};

} // namespace sare
