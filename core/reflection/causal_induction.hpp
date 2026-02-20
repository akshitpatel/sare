#pragma once

// ─── CausalInduction ──────────────────────────────────────────
// The "Scientist" engine. Tests candidate rules from ReflectionEngine
// *before* they are accepted into ConceptRegistry.
//
// Algorithm:
//   1. Take a candidate AbstractRule (pattern → replacement)
//   2. Generate N counter-examples by mutating the pattern
//   3. Attempt to verify the rule holds on each counter-example
//      (i.e., rule reduces energy when applied)
//   4. Compute P(rule valid | evidence)
//   5. Update rule.confidence accordingly
//   6. If confidence > ACCEPT_THRESHOLD → rule is promoted
//      If confidence < REJECT_THRESHOLD → rule is discarded
//
// This prevents the ConceptRegistry from learning false correlations.

#include "reflection_engine.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include <vector>
#include <string>
#include <random>

namespace sare {

struct InductionResult {
    bool promoted;              // Whether rule passed the hypothesis test
    double evidence_score;      // P(rule valid | evidence), [0,1]
    int tests_run;
    int tests_passed;
    std::string reasoning;      // Human-readable explanation of verdict
};

class CausalInduction {
public:
    // Thresholds for accept/reject
    static constexpr double ACCEPT_THRESHOLD = 0.65;
    static constexpr double REJECT_THRESHOLD = 0.30;
    static constexpr int    DEFAULT_TESTS     = 8;

    CausalInduction() : rng_(std::random_device{}()) {}

    /// Test a candidate rule and update its confidence score.
    /// \param rule   The candidate rule from ReflectionEngine (mutated in-place)
    /// \param energy Energy evaluator to measure rule quality
    /// \return Induction result with verdict and reasoning
    InductionResult evaluate(AbstractRule& rule,
                              const EnergyAggregator& energy,
                              int num_tests = DEFAULT_TESTS);

private:
    std::mt19937 rng_;

    /// Mutate a graph to create a counter-example for testing generalization.
    Graph generateCounterExample(const Graph& pattern);

    /// Check whether applying the rule to a test graph actually reduces energy.
    bool ruleAppliesCorrectly(const AbstractRule& rule,
                               const Graph& test_graph,
                               const EnergyAggregator& energy) const;

    /// Apply rule replacement to test_graph (pattern subgraph match).
    /// Returns empty optional if pattern doesn't match.
    std::optional<Graph> applyRule(const AbstractRule& rule,
                                    const Graph& test_graph) const;

    /// Simple structural subgraph match (type + label).
    bool subgraphMatch(const Graph& pattern,
                        const Graph& test_graph,
                        std::unordered_map<uint64_t,uint64_t>& mapping) const;
};

} // namespace sare
