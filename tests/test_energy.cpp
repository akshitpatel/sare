#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "energy/syntax_energy.hpp"
#include "energy/complexity_energy.hpp"
#include "energy/resource_energy.hpp"

using namespace sare;

// ─── Syntax Energy ─────────────────────────────────────────────

TEST(EnergyTest, SyntaxEnergyCleanGraph) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    uint64_t n2 = g.addNode("literal");
    g.addEdge(n1, n2, "references");

    SyntaxEnergy syntax;
    double e = syntax.compute(g);
    EXPECT_DOUBLE_EQ(e, 0.0);  // all nodes have types, all edges typed
}

TEST(EnergyTest, SyntaxEnergyErrorNode) {
    Graph g;
    g.addNode("error");

    SyntaxEnergy syntax;
    double e = syntax.compute(g);
    EXPECT_GT(e, 0.0);  // error type should be penalized
}

TEST(EnergyTest, SyntaxEnergyEmptyType) {
    Graph g;
    g.addNode("");  // empty type

    SyntaxEnergy syntax;
    double e = syntax.compute(g);
    EXPECT_GT(e, 0.0);
}

TEST(EnergyTest, SyntaxEnergyPerNode) {
    Graph g;
    uint64_t n1 = g.addNode("error");
    uint64_t n2 = g.addNode("variable");

    SyntaxEnergy syntax;
    EXPECT_GT(syntax.computeNode(g, n1), 0.0);
    EXPECT_DOUBLE_EQ(syntax.computeNode(g, n2), 0.0);
}

// ─── Complexity Energy ─────────────────────────────────────────

TEST(EnergyTest, ComplexityEnergySmallGraph) {
    Graph g;
    g.addNode("a");
    g.addNode("b");

    ComplexityEnergy complexity;
    double e = complexity.compute(g);
    EXPECT_GE(e, 0.0);
}

TEST(EnergyTest, ComplexityEnergyHighDegree) {
    Graph g;
    uint64_t center = g.addNode("hub");
    for (int i = 0; i < 15; i++) {
        uint64_t n = g.addNode("leaf");
        g.addEdge(center, n, "connects");
    }

    ComplexityEnergy complexity;
    double per_node = complexity.computeNode(g, center);
    EXPECT_GT(per_node, 0.0);  // high degree penalized
}

// ─── Energy Aggregator ─────────────────────────────────────────

TEST(EnergyTest, AggregatorCombinesComponents) {
    Graph g;
    g.addNode("error");  // will trigger syntax energy

    EnergyWeights weights;
    weights.alpha = 2.0;  // syntax weight

    EnergyAggregator aggregator(weights);
    aggregator.addComponent(std::make_unique<SyntaxEnergy>());

    auto breakdown = aggregator.computeTotal(g);
    EXPECT_GT(breakdown.syntax, 0.0);
    EXPECT_GT(breakdown.total(), 0.0);
}

TEST(EnergyTest, AggregatorEmptyGraph) {
    Graph g;

    EnergyAggregator aggregator;
    aggregator.addComponent(std::make_unique<SyntaxEnergy>());
    aggregator.addComponent(std::make_unique<ComplexityEnergy>());

    auto breakdown = aggregator.computeTotal(g);
    EXPECT_DOUBLE_EQ(breakdown.total(), 0.0);
}

// ─── Energy Breakdown Arithmetic ───────────────────────────────

TEST(EnergyTest, BreakdownAddition) {
    EnergyBreakdown a{1.0, 2.0, 0.0, 0.0, 0.0, 0.0};
    EnergyBreakdown b{0.0, 0.0, 3.0, 0.0, 0.0, 0.0};
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.syntax, 1.0);
    EXPECT_DOUBLE_EQ(c.constraint, 2.0);
    EXPECT_DOUBLE_EQ(c.test_failure, 3.0);
    EXPECT_DOUBLE_EQ(c.total(), 6.0);
}
