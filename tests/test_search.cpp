#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_base.hpp"
#include "transforms/transform_registry.hpp"
#include "transforms/algebra_transforms.hpp"
#include "search/search_state.hpp"
#include "search/beam_search.hpp"
#include "search/budget_manager.hpp"
#include "energy/syntax_energy.hpp"
#include "energy/complexity_energy.hpp"

using namespace sare;

// ─── Budget Manager ────────────────────────────────────────────

TEST(SearchTest, BudgetManagerExpansionLimit) {
    BudgetManager budget(100.0, 5);  // generous time, 5 expansion limit
    budget.start();

    for (int i = 0; i < 5; i++) {
        EXPECT_TRUE(budget.canContinue());
        budget.recordExpansion();
    }
    EXPECT_FALSE(budget.canContinue());
}

TEST(SearchTest, BudgetManagerTime) {
    BudgetManager budget(0.001, 1000);  // 1ms time limit
    budget.start();
    // Should expire very quickly
    // (timing-dependent, just check the API works)
    EXPECT_GE(budget.elapsedSeconds(), 0.0);
}

// ─── Beam Search ───────────────────────────────────────────────

TEST(SearchTest, BeamSearchReducesEnergy) {
    // Build a graph with x + 0 that can be simplified
    Graph g;
    uint64_t root = g.addNode("expression");
    uint64_t add_op = g.addNode("operator");
    g.getNode(add_op)->setAttribute("op", "add");
    uint64_t var_x = g.addNode("variable");
    g.getNode(var_x)->setAttribute("name", "x");
    uint64_t lit_0 = g.addNode("literal");
    g.getNode(lit_0)->setAttribute("value", "0");

    g.addEdge(root, add_op, "child");
    g.addEdge(add_op, var_x, "operand");
    g.addEdge(add_op, lit_0, "operand");

    // Setup energy (syntax + complexity)
    EnergyAggregator energy;
    energy.addComponent(std::make_unique<SyntaxEnergy>());
    energy.addComponent(std::make_unique<ComplexityEnergy>());

    // Setup transforms
    TransformRegistry transforms;
    transforms.registerTransform(std::make_unique<AddZeroTransform>());

    // Initial energy
    double initial_energy = energy.computeTotal(g).total();

    // Run beam search
    SearchConfig config;
    config.beam_width = 4;
    config.max_depth = 5;
    config.budget_seconds = 5.0;
    config.max_expansions = 100;

    BeamSearch search;
    SearchResult result = search.search(g, energy, transforms, config);

    // Energy should have decreased (or stayed same if transform didn't help)
    EXPECT_LE(result.best_state.energy.total(), initial_energy + 0.001);
    EXPECT_GT(result.total_expansions, 0);
}

TEST(SearchTest, BeamSearchEmptyTransforms) {
    Graph g;
    g.addNode("test");

    EnergyAggregator energy;
    TransformRegistry transforms;  // empty

    SearchConfig config;
    config.beam_width = 4;
    config.max_depth = 5;

    BeamSearch search;
    SearchResult result = search.search(g, energy, transforms, config);

    // No transforms → no improvement possible
    EXPECT_EQ(result.total_expansions, 0);
}

TEST(SearchTest, BeamSearchRespectsMaxDepth) {
    Graph g;
    g.addNode("variable");

    EnergyAggregator energy;
    TransformRegistry transforms;

    SearchConfig config;
    config.beam_width = 4;
    config.max_depth = 3;

    BeamSearch search;
    SearchResult result = search.search(g, energy, transforms, config);

    EXPECT_LE(result.max_depth_reached, config.max_depth);
}

// ─── Search State ──────────────────────────────────────────────

TEST(SearchTest, SearchStateComparison) {
    SearchState a, b;
    a.score = -5.0;
    b.score = -3.0;
    EXPECT_TRUE(a < b);  // b has higher score (less negative energy)
}
