#include "energy/constraint_energy.hpp"
#include <cmath>

namespace sare {

void ConstraintEnergy::addConstraint(ConstraintFn fn) {
    constraints_.push_back(std::move(fn));
}

double ConstraintEnergy::compute(const Graph& graph) const {
    double total = 0.0;
    graph.forEachNode([&](const Node& node) { total += computeNode(graph, node.id); });
    return total;
}

double ConstraintEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    double energy = 0.0;
    for (const auto& constraint : constraints_) {
        energy += constraint(graph, node_id);
    }
    return energy;
}

std::string ConstraintEnergy::name() const { return "constraint"; }

} // namespace sare
