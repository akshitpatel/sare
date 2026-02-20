#pragma once

#include "energy/energy.hpp"

namespace sare {

class SyntaxEnergy : public EnergyComponent {
public:
    double compute(const Graph& graph) const override;
    double computeNode(const Graph& graph, uint64_t node_id) const override;
    std::string name() const override;

private:
    double computeNodeEnergy(const Graph& graph, const Node& node) const;
};

} // namespace sare
