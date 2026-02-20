#pragma once

#include "energy/energy.hpp"
#include <functional>

namespace sare {

class ConstraintEnergy : public EnergyComponent {
public:
    using ConstraintFn = std::function<double(const Graph&, uint64_t)>;

    void addConstraint(ConstraintFn fn);
    double compute(const Graph& graph) const override;
    double computeNode(const Graph& graph, uint64_t node_id) const override;
    std::string name() const override;

private:
    std::vector<ConstraintFn> constraints_;
};

} // namespace sare
