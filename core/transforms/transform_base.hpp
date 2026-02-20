#pragma once

#include "graph/graph.hpp"
#include <string>

namespace sare {

/// Base class for all transformation modules.
/// M_k : G → ΔG
/// Each transform proposes a graph rewrite.
/// Must be pure (no side effects outside graph), reversible, and measurable.
class Transform {
public:
    virtual ~Transform() = default;

    /// Human-readable name of this transform.
    virtual std::string name() const = 0;

    /// Check if this transform is applicable to the current graph state.
    virtual bool match(const Graph& graph) const = 0;

    /// Apply the transform to the graph, returning the delta.
    /// Does NOT modify the graph directly — caller applies the delta.
    virtual GraphDelta apply(const Graph& graph) const = 0;

    /// Estimate the expected energy change (negative = improvement).
    virtual double estimateDeltaEnergy(const Graph& graph) const = 0;

    /// Computational cost of applying this transform.
    virtual double cost() const = 0;

    // ── Utility tracking ──

    /// Rolling utility: U_k = average(-ΔE) - cost
    double getUtility() const {
        if (application_count_ == 0) return 0.0;
        return (cumulative_delta_energy_ / application_count_) - cost();
    }

    void recordApplication(double actual_delta_energy) {
        cumulative_delta_energy_ += (-actual_delta_energy);  // negative ΔE = good
        application_count_++;
    }

    int applicationCount() const { return application_count_; }

private:
    double cumulative_delta_energy_ = 0.0;
    int application_count_ = 0;
};

} // namespace sare
