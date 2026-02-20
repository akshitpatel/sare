#include "energy/syntax_energy.hpp"

namespace sare {

double SyntaxEnergy::compute(const Graph& graph) const {
    double total = 0.0;
    graph.forEachNode([&](const Node& node) { total += computeNodeEnergy(graph, node); });
    return total;
}

double SyntaxEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    const Node* node = graph.getNode(node_id);
    if (!node) return 0.0;
    return computeNodeEnergy(graph, *node);
}

std::string SyntaxEnergy::name() const { return "syntax"; }

double SyntaxEnergy::computeNodeEnergy(const Graph& graph, const Node& node) const {
    double energy = 0.0;

    if (node.type.empty()) {
        energy += 1.0;
    }

    if (node.type == "error" || node.type == "undefined") {
        energy += 2.0;
    }

    auto outgoing = graph.getOutgoing(node.id);
    for (uint64_t eid : outgoing) {
        const Edge* e = graph.getEdge(eid);
        if (e && e->relationship_type.empty()) {
            energy += 0.5;
        }
    }

    return energy;
}

} // namespace sare
