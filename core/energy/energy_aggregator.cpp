#include "energy/energy.hpp"

namespace sare {

EnergyAggregator::EnergyAggregator(EnergyWeights weights)
    : weights_(weights) {}

void EnergyAggregator::addComponent(std::unique_ptr<EnergyComponent> component) {
    components_.push_back(std::move(component));
}

EnergyBreakdown EnergyAggregator::computeTotal(const Graph& graph) const {
    EnergyBreakdown bd;
    for (const auto& comp : components_) {
        double raw = comp->compute(graph);
        assignToBreakdown(bd, comp->name(), raw);
    }
    return bd;
}

EnergyBreakdown EnergyAggregator::computeNode(const Graph& graph, uint64_t node_id) const {
    EnergyBreakdown bd;
    for (const auto& comp : components_) {
        double raw = comp->computeNode(graph, node_id);
        assignToBreakdown(bd, comp->name(), raw);
    }
    return bd;
}

void EnergyAggregator::assignToBreakdown(EnergyBreakdown& bd,
                                          const std::string& name,
                                          double value) const {
    if (name == "syntax") {
        bd.syntax += weights_.alpha * value;
    } else if (name == "constraint") {
        bd.constraint += weights_.beta * value;
    } else if (name == "test_failure") {
        bd.test_failure += weights_.gamma * value;
    } else if (name == "complexity") {
        bd.complexity += weights_.delta * value;
    } else if (name == "resource") {
        bd.resource += weights_.lambda * value;
    } else if (name == "uncertainty") {
        bd.uncertainty += weights_.mu * value;
    }
}

} // namespace sare
