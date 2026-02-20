#include "causal/counterfactual.hpp"
#include <algorithm>
#include <cmath>

namespace sare {

CounterfactualResult CounterfactualSimulator::simulate(
    const Graph& g,
    const Intervention& intervention,
    const EnergyAggregator& energy) const {

    CounterfactualResult result;
    result.intervention = intervention;
    result.energy_original = energy.computeTotal(g).total();

    Graph cf_graph = engine_.doIntervention(g, intervention);
    result.energy_counterfactual = energy.computeTotal(cf_graph).total();
    result.delta = result.energy_counterfactual - result.energy_original;

    return result;
}

std::vector<CounterfactualResult> CounterfactualSimulator::compareInterventions(
    const Graph& g,
    const std::vector<Intervention>& interventions,
    const EnergyAggregator& energy) const {

    std::vector<CounterfactualResult> results;
    results.reserve(interventions.size());

    for (const auto& intervention : interventions) {
        results.push_back(simulate(g, intervention, energy));
    }

    // Sort by absolute delta (most impactful first)
    std::sort(results.begin(), results.end(),
              [](const CounterfactualResult& a, const CounterfactualResult& b) {
                  return std::abs(a.delta) > std::abs(b.delta);
              });

    return results;
}

CounterfactualResult CounterfactualSimulator::findMostImpactful(
    const Graph& g,
    const std::vector<Intervention>& interventions,
    const EnergyAggregator& energy) const {

    auto results = compareInterventions(g, interventions, energy);
    if (results.empty()) return CounterfactualResult{};
    return results[0];
}

} // namespace sare
