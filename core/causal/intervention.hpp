#pragma once

#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sare {

// ─── Intervention ─────────────────────────────────────────────
// Represents the causal do(X = x) operator.
// An intervention specifies which node/attribute to modify
// and what value to set.

struct Intervention {
    uint64_t node_id;
    std::string attribute;
    std::string value;
};

// ─── Intervention Engine ──────────────────────────────────────
// Implements the do() operator from causal inference.
// Forks the graph, applies the intervention, and evaluates
// the resulting energy to determine causal effects.

class InterventionEngine {
public:
    InterventionEngine() = default;

    /// Apply do(X = value) to a graph.
    /// Returns a new graph with the intervention applied.
    /// The original graph is not modified.
    Graph doIntervention(const Graph& g, const Intervention& intervention) const;

    /// Apply multiple interventions simultaneously.
    Graph doInterventions(const Graph& g,
                          const std::vector<Intervention>& interventions) const;

    /// Compare energy between original and intervened graph.
    double compareEnergies(const Graph& original,
                           const Graph& intervened,
                           const EnergyAggregator& energy) const;

    /// Check if an intervention has a causal effect (energy change > threshold).
    bool hasCausalEffect(const Graph& g, const Intervention& intervention,
                          const EnergyAggregator& energy,
                          double threshold = 0.01) const;
};

} // namespace sare
