#pragma once

#include "transforms/transform_base.hpp"
#include "transforms/transform_registry.hpp"
#include "abstraction/trace_miner.hpp"
#include <memory>
#include <vector>

namespace sare {

// ─── Macro Transform ──────────────────────────────────────────
// A composite transform built from a sequence of primitives.
// M_macro : G_P → G'_P
// Applies a chain of transforms as a single atomic operation.

class MacroTransform : public Transform {
public:
    MacroTransform(const std::string& macro_name,
                   const std::vector<std::string>& step_names)
        : name_(macro_name), step_names_(step_names) {}

    std::string name() const override { return name_; }

    /// Match: first step must match the current graph.
    bool match(const Graph& graph) const override;

    /// Apply: execute all steps sequentially, return combined delta.
    GraphDelta apply(const Graph& graph) const override;

    double estimateDeltaEnergy(const Graph& graph) const override;
    double cost() const override;

    /// Set the underlying transform pointers (looked up from registry).
    void bindSteps(const TransformRegistry& registry);

    size_t stepCount() const { return step_names_.size(); }
    const std::vector<std::string>& stepNames() const { return step_names_; }

private:
    std::string name_;
    std::vector<std::string> step_names_;
    std::vector<Transform*> steps_;  // non-owning pointers from registry
};

// ─── Macro Builder ────────────────────────────────────────────
// Constructs MacroTransforms from mined patterns.

class MacroBuilder {
public:
    /// Build a macro from a mined pattern.
    std::unique_ptr<MacroTransform> build(
        const TransformPattern& pattern,
        const TransformRegistry& registry);

    /// Auto-name based on pattern steps.
    static std::string generateName(const TransformPattern& pattern);
};

} // namespace sare
