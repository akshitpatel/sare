#include <gtest/gtest.h>
#include "search/mcts.hpp"
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_registry.hpp"
#include "transforms/transform_base.hpp"

using namespace sare;

// Simple transform for testing
class TestMCTSTransform : public Transform {
public:
    std::string name() const override { return "test_mcts_simplify"; }
    bool match(const Graph& g) const override {
        return g.nodeCount() > 2;
    }
    GraphDelta apply(const Graph& g) const override {
        GraphDelta delta;
        auto ids = g.getNodeIds();
        if (!ids.empty()) {
            delta.removed_node_ids.push_back(ids.back());
        }
        return delta;
    }
    double estimateDeltaEnergy(const Graph&) const override { return -1.0; }
    double cost() const override { return 0.1; }
};

class ComplexityEnergyForMCTS : public EnergyComponent {
public:
    double compute(const Graph& g) const override {
        return static_cast<double>(g.nodeCount()) * 2.0;
    }
    double computeNode(const Graph&, uint64_t) const override { return 2.0; }
    std::string name() const override { return "complexity"; }
};

TEST(MCTSTest, BasicSearch) {
    Graph g;
    g.addNode("variable");
    g.addNode("operator");
    g.addNode("constant");
    g.addNode("extra");

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<ComplexityEnergyForMCTS>());

    TransformRegistry registry;
    registry.registerTransform(std::make_unique<TestMCTSTransform>());

    SearchConfig config;
    config.max_depth = 5;
    config.max_expansions = 50;
    config.budget_seconds = 5.0;

    MCTSSearch mcts;
    SearchResult result = mcts.search(g, energy, registry, config);

    // MCTS should find a solution with lower energy
    EXPECT_LT(result.best_state.energy.total(), 8.0);  // < initial 4*2=8
    EXPECT_GT(result.total_expansions, 0);
}

TEST(MCTSTest, WithHeuristic) {
    Graph g;
    g.addNode("variable");
    g.addNode("operator");
    g.addNode("constant");

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<ComplexityEnergyForMCTS>());

    TransformRegistry registry;
    registry.registerTransform(std::make_unique<TestMCTSTransform>());

    SearchConfig config;
    config.max_depth = 3;
    config.max_expansions = 20;
    config.budget_seconds = 2.0;

    MCTSSearch mcts;
    // Set a simple heuristic: bonus for smaller graphs
    mcts.setHeuristic([](const Graph& g) { return 10.0 / (g.nodeCount() + 1); });

    SearchResult result = mcts.search(g, energy, registry, config);
    EXPECT_GT(result.total_expansions, 0);
}

TEST(MCTSTest, MCTSNodeUCB1) {
    MCTSNode node;
    node.visit_count = 10;
    node.total_reward = 50.0;  // avg = 5.0

    double ucb = node.ucb1(1.414, 100);
    // exploitation = 5.0, exploration = 1.414 * sqrt(ln(100)/10) ≈ 0.96
    EXPECT_GT(ucb, 5.0);
    EXPECT_LT(ucb, 7.0);
}

TEST(MCTSTest, UnvisitedNodeHighPriority) {
    MCTSNode node;
    node.visit_count = 0;

    double ucb = node.ucb1(1.414, 100);
    EXPECT_GT(ucb, 1e8);  // should be very high
}
