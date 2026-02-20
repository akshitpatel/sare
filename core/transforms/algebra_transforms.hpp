#pragma once

#include "transforms/transform_base.hpp"

namespace sare {

class AddZeroTransform : public Transform {
public:
    std::string name() const override;
    bool match(const Graph& graph) const override;
    GraphDelta apply(const Graph& graph) const override;
    double estimateDeltaEnergy(const Graph& graph) const override;
    double cost() const override;
};

class MulOneTransform : public Transform {
public:
    std::string name() const override;
    bool match(const Graph& graph) const override;
    GraphDelta apply(const Graph& graph) const override;
    double estimateDeltaEnergy(const Graph& graph) const override;
    double cost() const override;
};

class MulZeroTransform : public Transform {
public:
    std::string name() const override;
    bool match(const Graph& graph) const override;
    GraphDelta apply(const Graph& graph) const override;
    double estimateDeltaEnergy(const Graph& graph) const override;
    double cost() const override;
};

} // namespace sare
