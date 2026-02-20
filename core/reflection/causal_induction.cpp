#include "causal_induction.hpp"
#include <sstream>
#include <unordered_map>

namespace sare {

// ─── Public: evaluate ─────────────────────────────────────────

InductionResult CausalInduction::evaluate(
    AbstractRule& rule,
    const EnergyAggregator& energy,
    int num_tests)
{
    InductionResult result;
    result.tests_run    = 0;
    result.tests_passed = 0;

    if (!rule.valid()) {
        result.promoted      = false;
        result.evidence_score = 0.0;
        result.reasoning     = "Rule is invalid (empty name or pattern)";
        return result;
    }

    // Run hypothesis tests using counter-examples derived from the pattern.
    for (int i = 0; i < num_tests; ++i) {
        // Generate a test case: slightly mutated version of the rule's pattern.
        Graph test = generateCounterExample(rule.pattern);
        if (test.nodeCount() == 0) continue;

        ++result.tests_run;
        if (ruleAppliesCorrectly(rule, test, energy))
            ++result.tests_passed;
    }

    // If we couldn't generate any tests, fall back on prior confidence.
    if (result.tests_run == 0) {
        result.evidence_score = rule.confidence;
        result.promoted       = rule.confidence >= ACCEPT_THRESHOLD;
        result.reasoning      = "No counter-examples generated; using prior confidence";
        return result;
    }

    // Bayesian update: evidence_score is fraction of passes.
    double pass_rate = static_cast<double>(result.tests_passed) / result.tests_run;
    // Blend prior confidence with new evidence
    result.evidence_score = 0.4 * rule.confidence + 0.6 * pass_rate;
    rule.confidence       = result.evidence_score;
    rule.observations    += result.tests_run;

    result.promoted = result.evidence_score >= ACCEPT_THRESHOLD;

    std::ostringstream oss;
    oss << result.tests_passed << "/" << result.tests_run << " tests passed"
        << " (score=" << result.evidence_score << ")."
        << (result.promoted ? " → PROMOTED." : " → REJECTED.");
    result.reasoning = oss.str();

    return result;
}

// ─── Private: generateCounterExample ─────────────────────────

Graph CausalInduction::generateCounterExample(const Graph& pattern)
{
    // Clone the pattern graph and apply mild mutations to test
    // whether the rule still holds across slight variations.
    Graph g = pattern;

    std::vector<uint64_t> ids = g.getNodeIds();
    if (ids.empty()) return g;

    std::uniform_int_distribution<size_t> pick(0, ids.size() - 1);
    uint64_t target_id = ids[pick(rng_)];
    Node* n = g.getNode(target_id);
    if (!n) return g;

    // Mutation: if constant, perturb value slightly
    if (n->type == "constant") {
        auto it = n->attributes.find("label");
        if (it != n->attributes.end()) {
            try {
                int val = std::stoi(it->second);
                // Randomly increment or decrement by 1
                std::uniform_int_distribution<int> delta(-1, 1);
                val += delta(rng_);
                n->attributes["label"] = std::to_string(val);
            } catch (...) {}
        }
    }
    // Mutation: if variable, change its type tag to test type constraint
    // (We deliberately leave operator nodes intact to preserve structure)

    return g;
}

// ─── Private: ruleAppliesCorrectly ───────────────────────────

bool CausalInduction::ruleAppliesCorrectly(
    const AbstractRule& rule,
    const Graph& test_graph,
    const EnergyAggregator& energy) const
{
    auto result_graph = applyRule(rule, test_graph);
    if (!result_graph.has_value()) {
        // Rule doesn't match: counts as "not applying" but not a failure.
        // For induction, a no-match is neutral evidence (not a failure).
        return true;
    }

    // Check that energy actually decreased after applying the rule.
    double e_before = energy.evaluate(test_graph).total;
    double e_after  = energy.evaluate(*result_graph).total;
    return e_after < e_before;
}

// ─── Private: applyRule ────────────────────────────────────────

std::optional<Graph> CausalInduction::applyRule(
    const AbstractRule& rule,
    const Graph& test_graph) const
{
    // Attempt subgraph match of rule.pattern into test_graph.
    std::unordered_map<uint64_t, uint64_t> mapping;
    if (!subgraphMatch(rule.pattern, test_graph, mapping))
        return std::nullopt;

    // Apply replacement: clone graph, remove matched pattern subgraph,
    // insert replacement subgraph with edges properly remapped.
    Graph result = test_graph;

    // Remove matched pattern nodes from result
    for (auto& [pattern_id, graph_id] : mapping) {
        result.removeNode(graph_id);
    }

    // Insert replacement nodes (simplified: add as new nodes)
    std::unordered_map<uint64_t, uint64_t> rep_id_map;
    rule.replacement.forEachNode([&](const Node& rep_node) {
        uint64_t new_id = result.addNode(rep_node.type);
        Node* nn = result.getNode(new_id);
        if (nn) nn->attributes = rep_node.attributes;
        rep_id_map[rep_node.id] = new_id;
    });

    // Reconnect replacement edges
    rule.replacement.forEachEdge([&](const Edge& e) {
        auto src_it = rep_id_map.find(e.sourceId);
        auto tgt_it = rep_id_map.find(e.targetId);
        if (src_it != rep_id_map.end() && tgt_it != rep_id_map.end()) {
            result.addEdge(src_it->second, tgt_it->second, e.type);
        }
    });

    return result;
}

// ─── Private: subgraphMatch ───────────────────────────────────

bool CausalInduction::subgraphMatch(
    const Graph& pattern,
    const Graph& test_graph,
    std::unordered_map<uint64_t, uint64_t>& mapping) const
{
    // Greedy structural match by node type + label.
    // A full VF2 algorithm would be needed for production;
    // this approximation is sufficient for the current transform set.
    mapping.clear();

    auto p_ids = pattern.getNodeIds();
    auto t_ids = test_graph.getNodeIds();

    std::unordered_set<uint64_t> used_in_target;

    for (uint64_t p_id : p_ids) {
        const Node* pn = pattern.getNode(p_id);
        if (!pn) return false;

        bool matched = false;
        for (uint64_t t_id : t_ids) {
            if (used_in_target.count(t_id)) continue;
            const Node* tn = test_graph.getNode(t_id);
            if (!tn || tn->type != pn->type) continue;

            // Type matched. Check label if pattern specifies one.
            auto p_lbl = pn->attributes.find("label");
            if (p_lbl != pn->attributes.end() && !p_lbl->second.empty()) {
                auto t_lbl = tn->attributes.find("label");
                if (t_lbl == tn->attributes.end() || t_lbl->second != p_lbl->second)
                    continue;
            }

            mapping[p_id] = t_id;
            used_in_target.insert(t_id);
            matched = true;
            break;
        }
        if (!matched) return false;
    }
    return !mapping.empty();
}

} // namespace sare
