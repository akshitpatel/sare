#pragma once

#include "graph/graph.hpp"
#include <string>
#include <vector>
#include <memory>

namespace sare {

// ─── Energy Breakdown ──────────────────────────────────────────
// Stores per-component energy values. The total is the weighted sum.

struct EnergyBreakdown {
    double syntax       = 0.0;
    double constraint   = 0.0;
    double test_failure = 0.0;
    double complexity   = 0.0;
    double resource     = 0.0;
    double uncertainty  = 0.0;

    double total() const {
        return syntax + constraint + test_failure +
               complexity + resource + uncertainty;
    }

    EnergyBreakdown operator+(const EnergyBreakdown& other) const {
        return {
            syntax + other.syntax,
            constraint + other.constraint,
            test_failure + other.test_failure,
            complexity + other.complexity,
            resource + other.resource,
            uncertainty + other.uncertainty
        };
    }
};

// ─── Energy Component ──────────────────────────────────────────
// Abstract base class for individual energy terms.

class EnergyComponent {
public:
    virtual ~EnergyComponent() = default;

    /// Compute the energy for the entire graph.
    virtual double compute(const Graph& graph) const = 0;

    /// Compute the energy contribution of a single node.
    virtual double computeNode(const Graph& graph, uint64_t node_id) const = 0;

    /// Human-readable name of this component.
    virtual std::string name() const = 0;
};

// ─── Energy Weights ────────────────────────────────────────────

struct EnergyWeights {
    double alpha = 1.0;   // syntax
    double beta  = 1.0;   // constraint
    double gamma = 1.0;   // test failure
    double delta = 0.5;   // complexity
    double lambda = 0.3;  // resource
    double mu     = 0.2;  // uncertainty
};

// ─── Energy Aggregator ─────────────────────────────────────────
// Combines weighted energy components into a total energy score.
// Supports incremental recomputation via dirty-node tracking.

class EnergyAggregator {
public:
    explicit EnergyAggregator(EnergyWeights weights = {});

    void addComponent(std::unique_ptr<EnergyComponent> component);

    /// Compute total energy breakdown for the graph.
    EnergyBreakdown computeTotal(const Graph& graph) const;

    /// Compute energy contribution of a single node.
    EnergyBreakdown computeNode(const Graph& graph, uint64_t node_id) const;

    /// Get weights.
    const EnergyWeights& weights() const { return weights_; }
    void setWeights(const EnergyWeights& w) { weights_ = w; }

    size_t componentCount() const { return components_.size(); }

private:
    EnergyWeights weights_;
    std::vector<std::unique_ptr<EnergyComponent>> components_;

    // Map component name to field in EnergyBreakdown
    void assignToBreakdown(EnergyBreakdown& bd, const std::string& name, double value) const;
};

} // namespace sare
