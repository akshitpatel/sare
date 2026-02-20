#pragma once

#include "transforms/transform_base.hpp"

namespace sare {

class ConstantFoldTransform : public Transform {
public:
    std::string name() const override;
    bool match(const Graph& graph) const override;
    GraphDelta apply(const Graph& graph) const override;
    double estimateDeltaEnergy(const Graph& graph) const override;
    double cost() const override;
};

} // namespace sare
