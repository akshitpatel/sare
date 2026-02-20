#pragma once

#include "graph/graph.hpp"
#include "transforms/transform_base.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <optional>

namespace sare {

// ─── Causal Diff ──────────────────────────────────────────────
// Represents the structural difference between two graph states,
// focusing on the minimal causal mechanism of change.

struct ReflectionDiff {
    std::unordered_set<uint64_t> added_nodes;
    std::unordered_set<uint64_t> removed_nodes;
    std::unordered_set<uint64_t> modified_nodes;  // Attribute/edge changes
    std::unordered_set<uint64_t> context_nodes;   // Unchanged but connected neighbors
    bool empty() const {
        return added_nodes.empty() && removed_nodes.empty() && modified_nodes.empty();
    }
};

// ─── Type Constraint ─────────────────────────────────────────
// Represents the inferred type constraint on a pattern variable.

struct TypeConstraint {
    uint64_t node_id;          // Which pattern node this constrains
    std::string required_type; // e.g., "constant", "operator", "variable"
    std::string required_label; // e.g., "0", "+", "" (empty = any)
};

// ─── Abstract Rule ────────────────────────────────────────────
// A synthesized, generalized rule extracted from a solve instance.
// e.g., "Operator(+) with Right(constant:0) → Left"

struct AbstractRule {
    std::string name;                          // Human-readable: "additive_identity"
    std::string domain;                        // e.g., "arithmetic", "logic"
    Graph pattern;                             // The "before" subgraph template
    Graph replacement;                         // The "after" subgraph template
    std::vector<TypeConstraint> type_constraints; // Type constraints on variables
    double confidence = 0.5;
    int observations = 1;

    bool valid() const { return !name.empty() && pattern.nodeCount() > 0; }
};

// ─── Reflection Engine ────────────────────────────────────────
// The "Why" Engine. Analyzes solve traces to understand the
// causal mechanism that creates energy reduction.
//
// After every successful solve:
//   1. Compute the minimal diff (which nodes changed)
//   2. Extract type constraints from surrounding context
//   3. Synthesize a generalized rule with a semantically-meaningful name

class ReflectionEngine {
public:
    ReflectionEngine() = default;

    /// Analyze a successful solve to extract a general rule.
    /// \param initial The starting graph state
    /// \param final The final solved graph state
    /// \return A candidate abstract rule, or nullptr if no clear rule found
    std::unique_ptr<AbstractRule> reflect(const Graph& initial, const Graph& final_g);

    /// Compute the structural difference between two graphs.
    ReflectionDiff computeDiff(const Graph& initial, const Graph& final_g) const;

    /// Generalize a concrete diff into an abstract rule.
    /// Infers type constraints from the pattern context.
    std::unique_ptr<AbstractRule> generalize(
        const Graph& initial,
        const Graph& final_g,
        const ReflectionDiff& diff) const;

private:
    /// Infer a semantic name for the rule based on the pattern structure.
    std::string inferRuleName(const Graph& initial, const ReflectionDiff& diff) const;

    /// Infer domain from the types of nodes involved.
    std::string inferDomain(const Graph& pattern) const;

    /// Extract type constraints from pattern node context.
    std::vector<TypeConstraint> extractTypeConstraints(
        const Graph& initial,
        const std::unordered_set<uint64_t>& pattern_ids) const;
};

} // namespace sare
