#include <gtest/gtest.h>
#include "reflection/reflection_engine.hpp"

using namespace sare;

class ReflectionTest : public ::testing::Test {
protected:
    ReflectionEngine engine;
    Graph initial;
    Graph final;

    void SetUp() override {
        // Common setup if needed
    }
};

TEST_F(ReflectionTest, IdentityReflection) {
    // Scenario: initial graph has x + 0
    // final graph has x (0 and + removed)
    // x is preserved (context)

    // Initial: x --[left]--> + --[right]--> 0
    uint64_t x_id = initial.addNodeWithId(1, "variable"); // kept
    initial.getNode(x_id)->setAttribute("label", "x");
    
    uint64_t plus_id = initial.addNodeWithId(2, "operator"); // removed
    initial.getNode(plus_id)->setAttribute("label", "+");
    
    uint64_t zero_id = initial.addNodeWithId(3, "constant"); // removed
    initial.getNode(zero_id)->setAttribute("label", "0");

    initial.addEdge(plus_id, x_id, "left_operand");
    initial.addEdge(plus_id, zero_id, "right_operand");

    // Final: just x (same ID 1)
    final.addNodeWithId(1, "variable");
    final.getNode(1)->setAttribute("label", "x");

    auto rule = engine.reflect(initial, final);

    ASSERT_NE(rule, nullptr);
    
    // Pattern should contain +, 0, and x (context)
    EXPECT_EQ(rule->pattern.nodeCount(), 3);
    
    bool found_plus = false;
    bool found_zero = false;
    bool found_x = false;
    
    rule->pattern.forEachNode([&](const Node& n) {
        std::string label = n.getAttribute("label");
        if (label == "+") found_plus = true;
        if (label == "0") found_zero = true;
        if (label == "x") found_x = true;
    });

    EXPECT_TRUE(found_plus);
    EXPECT_TRUE(found_zero);
    EXPECT_TRUE(found_x);

    // Replacement should contain just x
    EXPECT_EQ(rule->replacement.nodeCount(), 1);
    EXPECT_EQ(rule->replacement.getNode(1)->getAttribute("label"), "x");
}

TEST_F(ReflectionTest, NoChange) {
    initial.addNode("A");
    final.addNode("A"); // different IDs but content same? No, IDs must match
    
    // Exact same graph
    uint64_t id = initial.addNode("A");
    final = initial.clone();

    auto rule = engine.reflect(initial, final);
    EXPECT_EQ(rule, nullptr);
}

TEST_F(ReflectionTest, Modification) {
    // Node A changes attribute
    uint64_t id = initial.addNodeWithId(10, "A");
    initial.getNode(id)->setAttribute("value", "1.0");

    final.addNodeWithId(10, "A");
    final.getNode(10)->setAttribute("value", "2.0");

    auto rule = engine.reflect(initial, final);
    ASSERT_NE(rule, nullptr);

    // Pattern has A(1.0)
    EXPECT_EQ(rule->pattern.nodeCount(), 1);
    EXPECT_EQ(rule->pattern.getNode(10)->getAttribute("value"), "1.0");

    // Replacement has A(2.0)
    EXPECT_EQ(rule->replacement.nodeCount(), 1);
    EXPECT_EQ(rule->replacement.getNode(10)->getAttribute("value"), "2.0");
}

TEST_F(ReflectionTest, ContextExtraction) {
    // A -> B. B is modified. A is context.
    uint64_t a = initial.addNodeWithId(1, "A");
    uint64_t b = initial.addNodeWithId(2, "B");
    initial.addEdge(1, 2, "link");
    initial.getNode(2)->setAttribute("val", "10");

    final.addNodeWithId(1, "A");
    final.addNodeWithId(2, "B");
    final.addEdge(1, 2, "link");
    final.getNode(2)->setAttribute("val", "20");

    auto rule = engine.reflect(initial, final);
    ASSERT_NE(rule, nullptr);

    // Pattern must include context A
    bool found_a_pat = false;
    rule->pattern.forEachNode([&](const Node& n) { if (n.type == "A") found_a_pat = true; });
    EXPECT_TRUE(found_a_pat);
    EXPECT_EQ(rule->pattern.nodeCount(), 2);

    // Replacement must include context A
    bool found_a_rep = false;
    rule->replacement.forEachNode([&](const Node& n) { if (n.type == "A") found_a_rep = true; });
    EXPECT_TRUE(found_a_rep);
    EXPECT_EQ(rule->replacement.nodeCount(), 2);
}
