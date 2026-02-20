#pragma once

#include "energy/energy.hpp"
#include <functional>

namespace sare {

class TestEnergy : public EnergyComponent {
public:
    using TestOracleFn = std::function<double(const Graph&, uint64_t)>;

    void setOracle(TestOracleFn fn);
    double compute(const Graph& graph) const override;
    double computeNode(const Graph& graph, uint64_t node_id) const override;
    std::string name() const override;

private:
    TestOracleFn oracle_;
};

} // namespace sare
