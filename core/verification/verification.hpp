#pragma once

#include "graph/graph.hpp"
#include <string>
#include <vector>

namespace sare {

/// Verification result for a single check.
struct VerificationResult {
    bool passed = false;
    std::string check_name;
    std::string message;
    uint64_t node_id = 0;  // 0 = graph-level
};

/// Syntax Checker: validates graph structural integrity.
class SyntaxChecker {
public:
    std::vector<VerificationResult> check(const Graph& graph) const;
};

/// Static Analyzer: validates domain-specific constraints.
class StaticAnalyzer {
public:
    using ConstraintFn = std::function<VerificationResult(const Graph&)>;

    void addConstraint(const std::string& name, ConstraintFn fn);
    std::vector<VerificationResult> analyze(const Graph& graph) const;

private:
    std::vector<std::pair<std::string, ConstraintFn>> constraints_;
};

/// Test case for the unit test runner.
struct TestCase {
    std::string name;
    std::function<bool(const Graph&)> test_fn;
};

/// Unit Test Runner: runs domain-specific tests against graph state.
class UnitTestRunner {
public:
    void addTest(const std::string& name, std::function<bool(const Graph&)> fn);
    std::vector<VerificationResult> run(const Graph& graph) const;

    size_t testCount() const { return tests_.size(); }

private:
    std::vector<TestCase> tests_;
};

} // namespace sare
