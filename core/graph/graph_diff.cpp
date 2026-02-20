#include "graph/graph_diff.hpp"
#include <unordered_set>

namespace sare {

GraphDelta GraphDiff::diff(const Graph& from, const Graph& to) {
    GraphDelta delta;

    // Nodes in `to` but not in `from` → added
    auto to_nodes = to.getNodeIds();
    auto from_nodes_set = std::unordered_set<uint64_t>();
    for (auto id : from.getNodeIds()) from_nodes_set.insert(id);
    auto to_nodes_set = std::unordered_set<uint64_t>();
    for (auto id : to_nodes) to_nodes_set.insert(id);

    for (uint64_t nid : to_nodes) {
        if (!from_nodes_set.count(nid)) {
            const Node* n = to.getNode(nid);
            if (n) delta.added_nodes.push_back(*n);
        }
    }

    // Nodes in `from` but not in `to` → removed
    for (uint64_t nid : from.getNodeIds()) {
        if (!to_nodes_set.count(nid)) {
            delta.removed_node_ids.push_back(nid);
            const Node* n = from.getNode(nid);
            if (n) {
                delta.removed_nodes.push_back(*n);
            }
        }
    }

    // Nodes in both → check for modifications
    for (uint64_t nid : from.getNodeIds()) {
        if (!to_nodes_set.count(nid)) continue;
        const Node* fn = from.getNode(nid);
        const Node* tn = to.getNode(nid);
        if (!fn || !tn) continue;

        bool modified = (fn->type != tn->type ||
                         fn->activation != tn->activation ||
                         fn->uncertainty != tn->uncertainty ||
                         fn->attributes != tn->attributes);
        if (modified) {
            delta.modified_nodes_before.push_back(*fn);
            delta.modified_nodes_after.push_back(*tn);
        }
    }

    // Same logic for edges
    auto to_edges = to.getEdgeIds();
    auto from_edges_set = std::unordered_set<uint64_t>();
    for (auto id : from.getEdgeIds()) from_edges_set.insert(id);
    auto to_edges_set = std::unordered_set<uint64_t>();
    for (auto id : to_edges) to_edges_set.insert(id);

    for (uint64_t eid : to_edges) {
        if (!from_edges_set.count(eid)) {
            const Edge* e = to.getEdge(eid);
            if (e) delta.added_edges.push_back(*e);
        }
    }

    for (uint64_t eid : from.getEdgeIds()) {
        if (!to_edges_set.count(eid)) {
            delta.removed_edge_ids.push_back(eid);
            const Edge* e = from.getEdge(eid);
            if (e) {
                delta.removed_edges.push_back(*e);
            }
        }
    }

    for (uint64_t eid : from.getEdgeIds()) {
        if (!to_edges_set.count(eid)) continue;
        const Edge* fe = from.getEdge(eid);
        const Edge* te = to.getEdge(eid);
        if (!fe || !te) continue;

        bool modified = (fe->relationship_type != te->relationship_type ||
                         fe->weight != te->weight);
        if (modified) {
            delta.modified_edges_before.push_back(*fe);
            delta.modified_edges_after.push_back(*te);
        }
    }

    return delta;
}

GraphDelta GraphDiff::invert(const GraphDelta& delta) {
    GraphDelta inv;

    inv.added_nodes = delta.removed_nodes;
    inv.added_edges = delta.removed_edges;

    for (const auto& n : delta.added_nodes) {
        inv.removed_node_ids.push_back(n.id);
        inv.removed_nodes.push_back(n);
    }
    for (const auto& e : delta.added_edges) {
        inv.removed_edge_ids.push_back(e.id);
        inv.removed_edges.push_back(e);
    }

    inv.modified_nodes_before = delta.modified_nodes_after;
    inv.modified_nodes_after = delta.modified_nodes_before;
    inv.modified_edges_before = delta.modified_edges_after;
    inv.modified_edges_after = delta.modified_edges_before;

    return inv;
}

} // namespace sare
