#pragma once

#include "energy/energy.hpp"

namespace sare {

class ResourceEnergy : public EnergyComponent {
public:
    void setComputeBudget(double budget);
    double compute(const Graph& graph) const override;
    double computeNode(const Graph& graph, uint64_t node_id) const override;
    std::string name() const override;

private:
    double budget_ = 0.0;
};

class UncertaintyEnergy : public EnergyComponent {
public:
    double compute(const Graph& graph) const override;
    double computeNode(const Graph& graph, uint64_t node_id) const override;
    std::string name() const override;

private:
    double computeNodeEnergy(const Node& node) const;
};

} // namespace sare
