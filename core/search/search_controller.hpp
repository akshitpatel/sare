#pragma once

#include "search/search_state.hpp"
#include "search/beam_search.hpp"
#include "search/mcts.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_registry.hpp"

#include <string>

namespace sare {

/// Search Controller: orchestrates search algorithms.
/// Currently uses BeamSearch. Will switch to hybrid Beam+MCTS in Phase 2.
class SearchController {
public:
    enum class Algorithm {
        BEAM_SEARCH,
        MCTS,           // Stubbed in Phase 1
        HYBRID          // Future: alternates between beam and MCTS
    };

    SearchController(Algorithm algo = Algorithm::BEAM_SEARCH)
        : algorithm_(algo) {}

    void setAlgorithm(Algorithm algo) { algorithm_ = algo; }

    /// Run the configured search algorithm.
    SearchResult run(
        const Graph& graph,
        const EnergyAggregator& energy,
        const TransformRegistry& transforms,
        const SearchConfig& config
    ) {
        switch (algorithm_) {
            case Algorithm::BEAM_SEARCH:
                return beam_.search(graph, energy, transforms, config);
            case Algorithm::MCTS:
                return mcts_.search(graph, energy, transforms, config);
            case Algorithm::HYBRID:
                // Phase 1: just use beam search
                return beam_.search(graph, energy, transforms, config);
        }
        return {};
    }

	private:
		Algorithm algorithm_;
		BeamSearch beam_;
		MCTSSearch mcts_;
	};

	} // namespace sare
