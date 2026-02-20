#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "transforms/transform_base.hpp"
#include "transforms/transform_registry.hpp"
#include "transforms/algebra_transforms.hpp"
#include "transforms/logic_transforms.hpp"
#include "transforms/ast_transforms.hpp"

using namespace sare;

// ─── Helper: build expression graph ───────────────────────────

Graph buildAddZeroGraph() {
    // Expression: x + 0
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
    return g;
}

Graph buildMulOneGraph() {
    // Expression: x * 1
    Graph g;
    uint64_t root = g.addNode("expression");
    uint64_t mul_op = g.addNode("operator");
    g.getNode(mul_op)->setAttribute("op", "mul");
    uint64_t var_x = g.addNode("variable");
    g.getNode(var_x)->setAttribute("name", "x");
    uint64_t lit_1 = g.addNode("literal");
    g.getNode(lit_1)->setAttribute("value", "1");

    g.addEdge(root, mul_op, "child");
    g.addEdge(mul_op, var_x, "operand");
    g.addEdge(mul_op, lit_1, "operand");
    return g;
}

Graph buildConstantExprGraph() {
    // Expression: 3 + 7
    Graph g;
    uint64_t add_op = g.addNode("operator");
    g.getNode(add_op)->setAttribute("op", "add");
    uint64_t lit_3 = g.addNode("literal");
    g.getNode(lit_3)->setAttribute("value", "3");
    uint64_t lit_7 = g.addNode("literal");
    g.getNode(lit_7)->setAttribute("value", "7");

    g.addEdge(add_op, lit_3, "operand");
    g.addEdge(add_op, lit_7, "operand");
    return g;
}

// ─── Algebra: x + 0 → x ──────────────────────────────────────

TEST(TransformTest, AddZeroMatches) {
    Graph g = buildAddZeroGraph();
    AddZeroTransform t;
    EXPECT_TRUE(t.match(g));
    EXPECT_EQ(t.name(), "algebra_add_zero");
}

TEST(TransformTest, AddZeroApplyReducesNodes) {
    Graph g = buildAddZeroGraph();
    size_t original_nodes = g.nodeCount();

    AddZeroTransform t;
    GraphDelta delta = t.apply(g);
    EXPECT_FALSE(delta.empty());

    g.applyDelta(delta);
    EXPECT_LT(g.nodeCount(), original_nodes);
}

TEST(TransformTest, AddZeroNoMatchOnCleanGraph) {
    Graph g;
    g.addNode("variable");
    AddZeroTransform t;
    EXPECT_FALSE(t.match(g));
}

// ─── Algebra: x * 1 → x ──────────────────────────────────────

TEST(TransformTest, MulOneMatches) {
    Graph g = buildMulOneGraph();
    MulOneTransform t;
    EXPECT_TRUE(t.match(g));
}

TEST(TransformTest, MulOneApplyReducesNodes) {
    Graph g = buildMulOneGraph();
    size_t original_nodes = g.nodeCount();

    MulOneTransform t;
    GraphDelta delta = t.apply(g);
    g.applyDelta(delta);
    EXPECT_LT(g.nodeCount(), original_nodes);
}

// ─── AST: Constant Folding ────────────────────────────────────

TEST(TransformTest, ConstantFoldMatches) {
    Graph g = buildConstantExprGraph();
    ConstantFoldTransform t;
    EXPECT_TRUE(t.match(g));
}

TEST(TransformTest, ConstantFoldComputes) {
    Graph g = buildConstantExprGraph();
    ConstantFoldTransform t;
    GraphDelta delta = t.apply(g);
    g.applyDelta(delta);

    // Should have collapsed to a single literal node
    EXPECT_EQ(g.nodeCount(), 1);
    const Node* result = g.getNode(1);  // operator node ID was 1
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->type, "literal");
    // Value should be "10.000000" (3 + 7)
    EXPECT_NE(result->getAttribute("value").find("10"), std::string::npos);
}

// ─── Transform Registry ──────────────────────────────────────

TEST(TransformTest, RegistryBasicOperation) {
    TransformRegistry registry;
    registry.registerTransform(std::make_unique<AddZeroTransform>());
    registry.registerTransform(std::make_unique<MulOneTransform>());
    registry.registerTransform(std::make_unique<ConstantFoldTransform>());

    EXPECT_EQ(registry.count(), 3);

    Graph g = buildAddZeroGraph();
    auto applicable = registry.getApplicable(g);
    EXPECT_GE(applicable.size(), 1);

    // Check lookup by name
    auto* t = registry.getByName("algebra_add_zero");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->name(), "algebra_add_zero");
}

TEST(TransformTest, UtilityTracking) {
    AddZeroTransform t;
    EXPECT_EQ(t.applicationCount(), 0);
    EXPECT_DOUBLE_EQ(t.getUtility(), 0.0);

    t.recordApplication(-1.0);  // energy decreased by 1.0
    EXPECT_EQ(t.applicationCount(), 1);
    EXPECT_GT(t.getUtility(), 0.0);  // should be positive (useful)
}

TEST(TransformTest, EstimateDeltaEnergy) {
    Graph g = buildAddZeroGraph();
    AddZeroTransform t;
    double estimate = t.estimateDeltaEnergy(g);
    EXPECT_LT(estimate, 0.0);  // simplification should reduce energy
}
