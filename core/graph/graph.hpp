#pragma once

#include "graph/node.hpp"
#include "graph/edge.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sare {

// ─── GraphDelta ────────────────────────────────────────────────
// Represents a set of mutations to a graph. Used by transforms,
// snapshots, and diff operations. Fully invertible.

struct GraphDelta {
    std::vector<Node> added_nodes;
    std::vector<Edge> added_edges;
    std::vector<Node> removed_nodes;
    std::vector<Edge> removed_edges;
    std::vector<uint64_t> removed_node_ids;
    std::vector<uint64_t> removed_edge_ids;

    // For undo: store original state of modified nodes/edges
    std::vector<Node> modified_nodes_before;
    std::vector<Node> modified_nodes_after;
    std::vector<Edge> modified_edges_before;
    std::vector<Edge> modified_edges_after;

    bool empty() const {
        return added_nodes.empty() && added_edges.empty() &&
               removed_nodes.empty() && removed_edges.empty() &&
               removed_node_ids.empty() && removed_edge_ids.empty() &&
               modified_nodes_after.empty() && modified_edges_after.empty();
    }
};

// ─── Graph ─────────────────────────────────────────────────────
// The Structured Cognitive State Graph. G = (N, E, T).
// Adjacency-list backed by unordered_maps for O(1) access.
// Supports subgraph extraction, delta application, and cloning.

class Graph {
public:
    Graph() = default;

    // ── Node operations ──
    uint64_t addNode(const std::string& type);
    uint64_t addNodeWithId(uint64_t id, const std::string& type);
    bool removeNode(uint64_t id);
    Node* getNode(uint64_t id);
    const Node* getNode(uint64_t id) const;
    std::vector<uint64_t> getNodeIds() const;
    size_t nodeCount() const { return nodes_.size(); }

    // ── Edge operations ──
    uint64_t addEdge(uint64_t source, uint64_t target,
                     const std::string& relationship_type, double weight = 1.0);
    uint64_t addEdgeWithId(uint64_t id, uint64_t source, uint64_t target,
                           const std::string& relationship_type, double weight = 1.0);
    bool removeEdge(uint64_t id);
    Edge* getEdge(uint64_t id);
    const Edge* getEdge(uint64_t id) const;
    std::vector<uint64_t> getEdgeIds() const;
    size_t edgeCount() const { return edges_.size(); }

    // ── Adjacency queries ──
    std::vector<uint64_t> getOutgoing(uint64_t node_id) const;
    std::vector<uint64_t> getIncoming(uint64_t node_id) const;
    std::vector<uint64_t> getNeighborNodes(uint64_t node_id) const;

    // ── Subgraph extraction ──
    Graph extractSubgraph(const std::unordered_set<uint64_t>& node_ids) const;

    // ── Delta operations ──
    void applyDelta(const GraphDelta& delta);
    void undoDelta(const GraphDelta& delta);

    // ── Cloning ──
    Graph clone() const;

    // ── Iteration ──
    void forEachNode(std::function<void(const Node&)> fn) const;
    void forEachEdge(std::function<void(const Edge&)> fn) const;

private:
    uint64_t next_node_id_ = 1;
    uint64_t next_edge_id_ = 1;

    std::unordered_map<uint64_t, Node> nodes_;
    std::unordered_map<uint64_t, Edge> edges_;

    // Adjacency lists: node_id → set of edge_ids
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> outgoing_;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> incoming_;
};

} // namespace sare
