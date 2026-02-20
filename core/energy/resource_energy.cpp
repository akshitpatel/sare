#include "energy/resource_energy.hpp"

namespace sare {

void ResourceEnergy::setComputeBudget(double budget) { budget_ = budget; }

double ResourceEnergy::compute(const Graph& graph) const {
    double estimated_compute =
        static_cast<double>(graph.nodeCount()) * 0.01 +
        static_cast<double>(graph.edgeCount()) * 0.005;

    if (budget_ > 0 && estimated_compute > budget_) {
        return (estimated_compute - budget_) * 10.0;
    }
    return 0.0;
}

double ResourceEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    auto out = graph.getOutgoing(node_id);
    return static_cast<double>(out.size()) * 0.005;
}

std::string ResourceEnergy::name() const { return "resource"; }

double UncertaintyEnergy::compute(const Graph& graph) const {
    double total = 0.0;
    graph.forEachNode([&](const Node& node) { total += computeNodeEnergy(node); });
    return total;
}

double UncertaintyEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    const Node* node = graph.getNode(node_id);
    if (!node) return 0.0;
    return computeNodeEnergy(*node);
}

std::string UncertaintyEnergy::name() const { return "uncertainty"; }

double UncertaintyEnergy::computeNodeEnergy(const Node& node) const {
    return node.uncertainty * node.uncertainty;
}

} // namespace sare
