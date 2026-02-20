#include "energy/complexity_energy.hpp"
#include <cmath>

namespace sare {

double ComplexityEnergy::compute(const Graph& graph) const {
    double energy = 0.0;

    size_t n = graph.nodeCount();
    if (n > 0) {
        energy += std::log2(static_cast<double>(n)) * 0.1;
    }

    size_t m = graph.edgeCount();
    if (n > 1) {
        double density = static_cast<double>(m) / (static_cast<double>(n) * (n - 1));
        if (density > 0.5) {
            energy += (density - 0.5) * 2.0;
        }
    }

    return energy;
}

double ComplexityEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    auto out = graph.getOutgoing(node_id);
    auto in = graph.getIncoming(node_id);
    double degree = static_cast<double>(out.size() + in.size());

    if (degree > 10.0) {
        return (degree - 10.0) * 0.1;
    }
    return 0.0;
}

std::string ComplexityEnergy::name() const { return "complexity"; }

} // namespace sare
