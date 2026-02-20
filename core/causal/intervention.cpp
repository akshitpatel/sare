#include "causal/intervention.hpp"
#include <cmath>

namespace sare {

Graph InterventionEngine::doIntervention(const Graph& g,
                                          const Intervention& intervention) const {
    Graph forked = g.clone();

    Node* node = forked.getNode(intervention.node_id);
    if (node) {
        node->attributes[intervention.attribute] = intervention.value;
        // Reset uncertainty for intervened variable (we know its value)
        node->uncertainty = 0.0;
    }

    return forked;
}

Graph InterventionEngine::doInterventions(
    const Graph& g,
    const std::vector<Intervention>& interventions) const {

    Graph forked = g.clone();

    for (const auto& intervention : interventions) {
        Node* node = forked.getNode(intervention.node_id);
        if (node) {
            node->attributes[intervention.attribute] = intervention.value;
            node->uncertainty = 0.0;
        }
    }

    return forked;
}

double InterventionEngine::compareEnergies(
    const Graph& original,
    const Graph& intervened,
    const EnergyAggregator& energy) const {

    double e_original = energy.computeTotal(original).total();
    double e_intervened = energy.computeTotal(intervened).total();
    return e_intervened - e_original;
}

bool InterventionEngine::hasCausalEffect(
    const Graph& g,
    const Intervention& intervention,
    const EnergyAggregator& energy,
    double threshold) const {

    Graph intervened = doIntervention(g, intervention);
    double delta = std::abs(compareEnergies(g, intervened, energy));
    return delta > threshold;
}

} // namespace sare
