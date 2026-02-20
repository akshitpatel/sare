#include "energy/test_energy.hpp"

namespace sare {

void TestEnergy::setOracle(TestOracleFn fn) {
    oracle_ = std::move(fn);
}

double TestEnergy::compute(const Graph& graph) const {
    if (!oracle_) return 0.0;
    double total = 0.0;
    graph.forEachNode([&](const Node& node) { total += computeNode(graph, node.id); });
    return total;
}

double TestEnergy::computeNode(const Graph& graph, uint64_t node_id) const {
    if (!oracle_) return 0.0;
    return oracle_(graph, node_id);
}

std::string TestEnergy::name() const { return "test_failure"; }

} // namespace sare
