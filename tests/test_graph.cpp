#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "graph/subgraph_matcher.hpp"
#include "graph/graph_snapshot.hpp"
#include "graph/graph_diff.hpp"

using namespace sare;

// ─── Basic Node/Edge CRUD ──────────────────────────────────────

TEST(GraphTest, AddAndGetNode) {
    Graph g;
    uint64_t id = g.addNode("variable");
    ASSERT_EQ(g.nodeCount(), 1);
    const Node* n = g.getNode(id);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->type, "variable");
}

TEST(GraphTest, AddNodeWithAttributes) {
    Graph g;
    uint64_t id = g.addNode("literal");
    Node* n = g.getNode(id);
    n->setAttribute("value", "42");
    EXPECT_EQ(n->getAttribute("value"), "42");
    EXPECT_TRUE(n->hasAttribute("value"));
    EXPECT_FALSE(n->hasAttribute("nonexistent"));
}

TEST(GraphTest, RemoveNode) {
    Graph g;
    uint64_t id = g.addNode("test");
    ASSERT_TRUE(g.removeNode(id));
    EXPECT_EQ(g.nodeCount(), 0);
    EXPECT_EQ(g.getNode(id), nullptr);
    EXPECT_FALSE(g.removeNode(id));  // already removed
}

TEST(GraphTest, AddAndGetEdge) {
    Graph g;
    uint64_t n1 = g.addNode("a");
    uint64_t n2 = g.addNode("b");
    uint64_t eid = g.addEdge(n1, n2, "connects", 0.5);
    ASSERT_EQ(g.edgeCount(), 1);
    const Edge* e = g.getEdge(eid);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->source, n1);
    EXPECT_EQ(e->target, n2);
    EXPECT_EQ(e->relationship_type, "connects");
    EXPECT_DOUBLE_EQ(e->weight, 0.5);
}

TEST(GraphTest, RemoveEdge) {
    Graph g;
    uint64_t n1 = g.addNode("a");
    uint64_t n2 = g.addNode("b");
    uint64_t eid = g.addEdge(n1, n2, "connects");
    ASSERT_TRUE(g.removeEdge(eid));
    EXPECT_EQ(g.edgeCount(), 0);
}

TEST(GraphTest, RemoveNodeCascadesEdges) {
    Graph g;
    uint64_t n1 = g.addNode("a");
    uint64_t n2 = g.addNode("b");
    g.addEdge(n1, n2, "connects");
    g.removeNode(n1);
    EXPECT_EQ(g.nodeCount(), 1);
    EXPECT_EQ(g.edgeCount(), 0);
}

// ─── Adjacency Queries ────────────────────────────────────────

TEST(GraphTest, AdjacencyQueries) {
    Graph g;
    uint64_t n1 = g.addNode("a");
    uint64_t n2 = g.addNode("b");
    uint64_t n3 = g.addNode("c");
    g.addEdge(n1, n2, "to");
    g.addEdge(n1, n3, "to");
    g.addEdge(n3, n1, "back");

    auto out = g.getOutgoing(n1);
    EXPECT_EQ(out.size(), 2);

    auto in = g.getIncoming(n1);
    EXPECT_EQ(in.size(), 1);

    auto neighbors = g.getNeighborNodes(n1);
    EXPECT_EQ(neighbors.size(), 2);  // n2, n3
}

// ─── Subgraph Extraction ──────────────────────────────────────

TEST(GraphTest, ExtractSubgraph) {
    Graph g;
    uint64_t n1 = g.addNode("a");
    uint64_t n2 = g.addNode("b");
    uint64_t n3 = g.addNode("c");
    g.addEdge(n1, n2, "ab");
    g.addEdge(n2, n3, "bc");

    Graph sub = g.extractSubgraph({n1, n2});
    EXPECT_EQ(sub.nodeCount(), 2);
    EXPECT_EQ(sub.edgeCount(), 1);  // only ab edge
}

// ─── Clone ─────────────────────────────────────────────────────

TEST(GraphTest, Clone) {
    Graph g;
    uint64_t n1 = g.addNode("test");
    g.getNode(n1)->setAttribute("key", "val");

    Graph copy = g.clone();
    EXPECT_EQ(copy.nodeCount(), 1);
    EXPECT_EQ(copy.getNode(n1)->getAttribute("key"), "val");

    // Mutations should be independent
    copy.addNode("new");
    EXPECT_EQ(g.nodeCount(), 1);
    EXPECT_EQ(copy.nodeCount(), 2);
}

// ─── Delta Operations ──────────────────────────────────────────

TEST(GraphTest, ApplyDelta) {
    Graph g;
    uint64_t n1 = g.addNode("existing");

    GraphDelta delta;
    Node new_node(100, "added");
    delta.added_nodes.push_back(new_node);

    g.applyDelta(delta);
    EXPECT_EQ(g.nodeCount(), 2);
    EXPECT_NE(g.getNode(100), nullptr);
}

TEST(GraphTest, ModifyDelta) {
    Graph g;
    uint64_t n1 = g.addNode("before");
    Node* n = g.getNode(n1);
    n->setAttribute("val", "old");

    GraphDelta delta;
    Node before_state = *n;
    Node after_state = *n;
    after_state.setAttribute("val", "new");
    delta.modified_nodes_before.push_back(before_state);
    delta.modified_nodes_after.push_back(after_state);

    g.applyDelta(delta);
    EXPECT_EQ(g.getNode(n1)->getAttribute("val"), "new");

    // Undo
    g.undoDelta(delta);
    EXPECT_EQ(g.getNode(n1)->getAttribute("val"), "old");
}

// ─── Graph Diff ────────────────────────────────────────────────

TEST(GraphDiffTest, DiffDetectsAddedNode) {
    Graph a, b;
    a.addNode("same");
    b.addNodeWithId(1, "same");
    b.addNodeWithId(2, "new");

    auto delta = GraphDiff::diff(a, b);
    EXPECT_EQ(delta.added_nodes.size(), 1);
    EXPECT_EQ(delta.added_nodes[0].type, "new");
}

TEST(GraphDiffTest, DiffDetectsRemovedNode) {
    Graph a, b;
    a.addNode("will_remove");
    a.addNode("stays");
    b.addNodeWithId(2, "stays");

    auto delta = GraphDiff::diff(a, b);
    EXPECT_EQ(delta.removed_node_ids.size(), 1);
}

TEST(GraphDiffTest, DiffApplyInvertRoundTrip) {
    Graph from;
    uint64_t n1 = from.addNodeWithId(1, "operator");
    uint64_t n2 = from.addNodeWithId(2, "literal");
    from.getNode(n1)->setAttribute("op", "add");
    from.getNode(n2)->setAttribute("value", "0");
    from.addEdgeWithId(10, n1, n2, "operand", 1.0);

    Graph to = from.clone();
    to.removeNode(2);
    uint64_t n3 = to.addNodeWithId(3, "variable");
    to.getNode(n3)->setAttribute("name", "x");
    to.getNode(1)->setAttribute("op", "mul");
    to.addEdgeWithId(11, 1, 3, "operand", 1.0);

    auto same_graph = [](const Graph& lhs, const Graph& rhs) {
        if (lhs.nodeCount() != rhs.nodeCount() || lhs.edgeCount() != rhs.edgeCount()) {
            return false;
        }
        for (uint64_t nid : lhs.getNodeIds()) {
            const Node* ln = lhs.getNode(nid);
            const Node* rn = rhs.getNode(nid);
            if (!ln || !rn) return false;
            if (ln->type != rn->type || ln->attributes != rn->attributes) return false;
        }
        for (uint64_t eid : lhs.getEdgeIds()) {
            const Edge* le = lhs.getEdge(eid);
            const Edge* re = rhs.getEdge(eid);
            if (!le || !re) return false;
            if (le->source != re->source || le->target != re->target ||
                le->relationship_type != re->relationship_type) {
                return false;
            }
        }
        return true;
    };

    GraphDelta delta = GraphDiff::diff(from, to);
    ASSERT_FALSE(delta.empty());
    ASSERT_FALSE(delta.removed_nodes.empty());
    ASSERT_FALSE(delta.removed_edges.empty());

    Graph forward = from.clone();
    forward.applyDelta(delta);
    EXPECT_TRUE(same_graph(forward, to));

    GraphDelta inv = GraphDiff::invert(delta);
    Graph backward = to.clone();
    backward.applyDelta(inv);
    EXPECT_TRUE(same_graph(backward, from));

    Graph undo_case = from.clone();
    undo_case.applyDelta(delta);
    undo_case.undoDelta(delta);
    EXPECT_TRUE(same_graph(undo_case, from));
}

// ─── Subgraph Matcher ─────────────────────────────────────────

TEST(SubgraphMatcherTest, SimplePatternMatch) {
    Graph g;
    uint64_t n1 = g.addNode("operator");
    uint64_t n2 = g.addNode("literal");
    g.addEdge(n1, n2, "operand");

    MatchPattern pattern;
    pattern.nodes.push_back({1, "operator", {}});
    pattern.nodes.push_back({2, "literal", {}});
    pattern.edges.push_back({1, 2, "operand"});

    auto results = SubgraphMatcher::match(pattern, g);
    EXPECT_GE(results.size(), 1);
}

TEST(SubgraphMatcherTest, AttributeConstraint) {
    Graph g;
    uint64_t n1 = g.addNode("literal");
    g.getNode(n1)->setAttribute("value", "42");
    uint64_t n2 = g.addNode("literal");
    g.getNode(n2)->setAttribute("value", "0");

    MatchPattern pattern;
    pattern.nodes.push_back({1, "literal", {{"value", "0"}}});

    auto results = SubgraphMatcher::match(pattern, g);
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].at(1), n2);
}

// ─── Snapshot Manager ──────────────────────────────────────────

TEST(SnapshotTest, TakeAndRestore) {
    Graph root;
    uint64_t n1 = root.addNode("root_node");

    SnapshotManager manager;

    // Create a delta: add a node
    GraphDelta delta1;
    Node new_node(100, "snapshot_node");
    delta1.added_nodes.push_back(new_node);

    uint64_t snap1 = manager.takeSnapshot(delta1, 0);  // parent = root
    EXPECT_EQ(manager.snapshotCount(), 1);

    Graph restored = manager.restore(root, snap1);
    EXPECT_EQ(restored.nodeCount(), 2);
    EXPECT_NE(restored.getNode(100), nullptr);
}
