#pragma once

#include "causal/intervention.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include <vector>

namespace sare {

// ─── Counterfactual Result ────────────────────────────────────
// Result of a single counterfactual simulation.

struct CounterfactualResult {
    Intervention intervention;
    double energy_original = 0.0;
    double energy_counterfactual = 0.0;
    double delta = 0.0;  // E_cf - E_original
};

// ─── Counterfactual Simulator ─────────────────────────────────
// Implements counterfactual reasoning:
// "What would happen if X had been different?"
//
// Process:
// 1. Fork graph
// 2. Apply intervention do(X = x)
// 3. Evaluate counterfactual energy E_cf(G')
// 4. Compare with original energy

class CounterfactualSimulator {
public:
    CounterfactualSimulator() = default;

    /// Simulate a single counterfactual.
    CounterfactualResult simulate(
        const Graph& g,
        const Intervention& intervention,
        const EnergyAggregator& energy) const;

    /// Compare multiple interventions on the same graph.
    /// Returns results sorted by delta (most impactful first).
    std::vector<CounterfactualResult> compareInterventions(
        const Graph& g,
        const std::vector<Intervention>& interventions,
        const EnergyAggregator& energy) const;

    /// Find the most impactful intervention from a set.
    CounterfactualResult findMostImpactful(
        const Graph& g,
        const std::vector<Intervention>& interventions,
        const EnergyAggregator& energy) const;

private:
    InterventionEngine engine_;
};

} // namespace sare
