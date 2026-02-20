#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "verification/verification.hpp"

using namespace sare;

// ─── Syntax Checker ────────────────────────────────────────────

TEST(VerificationTest, SyntaxCheckerCleanGraph) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    uint64_t n2 = g.addNode("literal");
    g.addEdge(n1, n2, "references");

    SyntaxChecker checker;
    auto results = checker.check(g);
    ASSERT_EQ(results.size(), 1);
    EXPECT_TRUE(results[0].passed);
}

TEST(VerificationTest, SyntaxCheckerEmptyType) {
    Graph g;
    g.addNode("");  // empty type

    SyntaxChecker checker;
    auto results = checker.check(g);
    bool has_failure = false;
    for (const auto& r : results) {
        if (!r.passed) has_failure = true;
    }
    EXPECT_TRUE(has_failure);
}

TEST(VerificationTest, SyntaxCheckerErrorNode) {
    Graph g;
    g.addNode("error");

    SyntaxChecker checker;
    auto results = checker.check(g);
    bool has_failure = false;
    for (const auto& r : results) {
        if (!r.passed) has_failure = true;
    }
    EXPECT_TRUE(has_failure);
}

// ─── Unit Test Runner ──────────────────────────────────────────

TEST(VerificationTest, UnitTestRunnerAllPass) {
    Graph g;
    g.addNode("variable");

    UnitTestRunner runner;
    runner.addTest("has_nodes", [](const Graph& g) { return g.nodeCount() > 0; });
    runner.addTest("no_errors", [](const Graph& g) {
        bool clean = true;
        g.forEachNode([&](const Node& n) { if (n.type == "error") clean = false; });
        return clean;
    });

    auto results = runner.run(g);
    ASSERT_EQ(results.size(), 2);
    for (const auto& r : results) {
        EXPECT_TRUE(r.passed);
    }
}

TEST(VerificationTest, UnitTestRunnerWithFailure) {
    Graph g;
    g.addNode("error");

    UnitTestRunner runner;
    runner.addTest("no_errors", [](const Graph& g) {
        bool clean = true;
        g.forEachNode([&](const Node& n) { if (n.type == "error") clean = false; });
        return clean;
    });

    auto results = runner.run(g);
    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].passed);
}

// ─── Static Analyzer ──────────────────────────────────────────

TEST(VerificationTest, StaticAnalyzerCustomConstraint) {
    Graph g;
    g.addNode("variable");

    StaticAnalyzer analyzer;
    analyzer.addConstraint("max_nodes", [](const Graph& g) {
        VerificationResult r;
        r.check_name = "max_nodes";
        r.passed = g.nodeCount() <= 100;
        r.message = r.passed ? "Node count OK" : "Too many nodes";
        return r;
    });

    auto results = analyzer.analyze(g);
    ASSERT_EQ(results.size(), 1);
    EXPECT_TRUE(results[0].passed);
}
