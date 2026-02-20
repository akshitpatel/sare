#include "verification/verification.hpp"

namespace sare {

// ─── Syntax Checker ────────────────────────────────────────────

std::vector<VerificationResult> SyntaxChecker::check(const Graph& graph) const {
    std::vector<VerificationResult> results;

    graph.forEachNode([&](const Node& node) {
        // Check: node must have a type
        if (node.type.empty()) {
            results.push_back({false, "node_has_type",
                "Node " + std::to_string(node.id) + " has empty type", node.id});
        }

        // Check: error/undefined nodes are syntax violations
        if (node.type == "error" || node.type == "undefined") {
            results.push_back({false, "node_valid_type",
                "Node " + std::to_string(node.id) + " has error type: " + node.type, node.id});
        }
    });

    graph.forEachEdge([&](const Edge& edge) {
        // Check: edge endpoints must exist
        if (!graph.getNode(edge.source)) {
            results.push_back({false, "edge_source_exists",
                "Edge " + std::to_string(edge.id) + " has missing source " +
                std::to_string(edge.source), 0});
        }
        if (!graph.getNode(edge.target)) {
            results.push_back({false, "edge_target_exists",
                "Edge " + std::to_string(edge.id) + " has missing target " +
                std::to_string(edge.target), 0});
        }

        // Check: edge must have relationship type
        if (edge.relationship_type.empty()) {
            results.push_back({false, "edge_has_type",
                "Edge " + std::to_string(edge.id) + " has empty relationship type", 0});
        }
    });

    // If no failures, add a passing result
    if (results.empty()) {
        results.push_back({true, "syntax_check", "All syntax checks passed", 0});
    }

    return results;
}

// ─── Static Analyzer ───────────────────────────────────────────

void StaticAnalyzer::addConstraint(const std::string& name, ConstraintFn fn) {
    constraints_.push_back({name, std::move(fn)});
}

std::vector<VerificationResult> StaticAnalyzer::analyze(const Graph& graph) const {
    std::vector<VerificationResult> results;
    for (const auto& [name, fn] : constraints_) {
        results.push_back(fn(graph));
    }
    return results;
}

// ─── Unit Test Runner ──────────────────────────────────────────

void UnitTestRunner::addTest(const std::string& name, std::function<bool(const Graph&)> fn) {
    tests_.push_back({name, std::move(fn)});
}

std::vector<VerificationResult> UnitTestRunner::run(const Graph& graph) const {
    std::vector<VerificationResult> results;
    for (const auto& test : tests_) {
        bool passed = test.test_fn(graph);
        results.push_back({passed, test.name,
            passed ? "PASS" : "FAIL", 0});
    }
    return results;
}

} // namespace sare
