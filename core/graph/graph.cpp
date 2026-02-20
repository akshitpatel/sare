#include "graph/graph.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace sare {

// ─── Node operations ───────────────────────────────────────────

uint64_t Graph::addNode(const std::string& type) {
    uint64_t id = next_node_id_++;
    nodes_.emplace(id, Node(id, type));
    outgoing_[id];  // ensure entry exists
    incoming_[id];
    return id;
}

uint64_t Graph::addNodeWithId(uint64_t id, const std::string& type) {
    if (nodes_.count(id)) {
        throw std::runtime_error("Node ID already exists: " + std::to_string(id));
    }
    nodes_.emplace(id, Node(id, type));
    outgoing_[id];
    incoming_[id];
    if (id >= next_node_id_) {
        next_node_id_ = id + 1;
    }
    return id;
}

bool Graph::removeNode(uint64_t id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;

    // Remove all connected edges
    std::vector<uint64_t> edges_to_remove;
    if (outgoing_.count(id)) {
        for (auto eid : outgoing_[id]) edges_to_remove.push_back(eid);
    }
    if (incoming_.count(id)) {
        for (auto eid : incoming_[id]) edges_to_remove.push_back(eid);
    }
    for (auto eid : edges_to_remove) {
        removeEdge(eid);
    }

    outgoing_.erase(id);
    incoming_.erase(id);
    nodes_.erase(it);
    return true;
}

Node* Graph::getNode(uint64_t id) {
    auto it = nodes_.find(id);
    return it != nodes_.end() ? &it->second : nullptr;
}

const Node* Graph::getNode(uint64_t id) const {
    auto it = nodes_.find(id);
    return it != nodes_.end() ? &it->second : nullptr;
}

std::vector<uint64_t> Graph::getNodeIds() const {
    std::vector<uint64_t> ids;
    ids.reserve(nodes_.size());
    for (const auto& [id, _] : nodes_) {
        ids.push_back(id);
    }
    return ids;
}

// ─── Edge operations ───────────────────────────────────────────

uint64_t Graph::addEdge(uint64_t source, uint64_t target,
                        const std::string& relationship_type, double weight) {
    if (!nodes_.count(source))
        throw std::runtime_error("Source node not found: " + std::to_string(source));
    if (!nodes_.count(target))
        throw std::runtime_error("Target node not found: " + std::to_string(target));

    uint64_t id = next_edge_id_++;
    edges_.emplace(id, Edge(id, source, target, relationship_type, weight));
    outgoing_[source].insert(id);
    incoming_[target].insert(id);
    return id;
}

uint64_t Graph::addEdgeWithId(uint64_t id, uint64_t source, uint64_t target,
                              const std::string& relationship_type, double weight) {
    if (edges_.count(id))
        throw std::runtime_error("Edge ID already exists: " + std::to_string(id));
    if (!nodes_.count(source))
        throw std::runtime_error("Source node not found: " + std::to_string(source));
    if (!nodes_.count(target))
        throw std::runtime_error("Target node not found: " + std::to_string(target));

    edges_.emplace(id, Edge(id, source, target, relationship_type, weight));
    outgoing_[source].insert(id);
    incoming_[target].insert(id);
    if (id >= next_edge_id_) {
        next_edge_id_ = id + 1;
    }
    return id;
}

bool Graph::removeEdge(uint64_t id) {
    auto it = edges_.find(id);
    if (it == edges_.end()) return false;

    const Edge& e = it->second;
    if (outgoing_.count(e.source)) outgoing_[e.source].erase(id);
    if (incoming_.count(e.target)) incoming_[e.target].erase(id);

    edges_.erase(it);
    return true;
}

Edge* Graph::getEdge(uint64_t id) {
    auto it = edges_.find(id);
    return it != edges_.end() ? &it->second : nullptr;
}

const Edge* Graph::getEdge(uint64_t id) const {
    auto it = edges_.find(id);
    return it != edges_.end() ? &it->second : nullptr;
}

std::vector<uint64_t> Graph::getEdgeIds() const {
    std::vector<uint64_t> ids;
    ids.reserve(edges_.size());
    for (const auto& [id, _] : edges_) {
        ids.push_back(id);
    }
    return ids;
}

// ─── Adjacency queries ────────────────────────────────────────

std::vector<uint64_t> Graph::getOutgoing(uint64_t node_id) const {
    auto it = outgoing_.find(node_id);
    if (it == outgoing_.end()) return {};
    return std::vector<uint64_t>(it->second.begin(), it->second.end());
}

std::vector<uint64_t> Graph::getIncoming(uint64_t node_id) const {
    auto it = incoming_.find(node_id);
    if (it == incoming_.end()) return {};
    return std::vector<uint64_t>(it->second.begin(), it->second.end());
}

std::vector<uint64_t> Graph::getNeighborNodes(uint64_t node_id) const {
    std::unordered_set<uint64_t> neighbors;
    for (auto eid : getOutgoing(node_id)) {
        const Edge* e = getEdge(eid);
        if (e) neighbors.insert(e->target);
    }
    for (auto eid : getIncoming(node_id)) {
        const Edge* e = getEdge(eid);
        if (e) neighbors.insert(e->source);
    }
    return std::vector<uint64_t>(neighbors.begin(), neighbors.end());
}

// ─── Subgraph extraction ──────────────────────────────────────

Graph Graph::extractSubgraph(const std::unordered_set<uint64_t>& node_ids) const {
    Graph sub;
    for (uint64_t nid : node_ids) {
        const Node* n = getNode(nid);
        if (!n) continue;
        sub.addNodeWithId(n->id, n->type);
        Node* sn = sub.getNode(n->id);
        sn->attributes = n->attributes;
        sn->activation = n->activation;
        sn->uncertainty = n->uncertainty;
        sn->metadata = n->metadata;
    }
    for (const auto& [eid, edge] : edges_) {
        if (node_ids.count(edge.source) && node_ids.count(edge.target)) {
            sub.addEdgeWithId(edge.id, edge.source, edge.target,
                              edge.relationship_type, edge.weight);
        }
    }
    return sub;
}

// ─── Delta operations ──────────────────────────────────────────

void Graph::applyDelta(const GraphDelta& delta) {
    std::unordered_set<uint64_t> edges_to_remove;
    edges_to_remove.insert(delta.removed_edge_ids.begin(), delta.removed_edge_ids.end());
    for (const Edge& e : delta.removed_edges) {
        edges_to_remove.insert(e.id);
    }
    for (uint64_t eid : edges_to_remove) {
        removeEdge(eid);
    }

    std::unordered_set<uint64_t> nodes_to_remove;
    nodes_to_remove.insert(delta.removed_node_ids.begin(), delta.removed_node_ids.end());
    for (const Node& n : delta.removed_nodes) {
        nodes_to_remove.insert(n.id);
    }
    for (uint64_t nid : nodes_to_remove) {
        removeNode(nid);
    }

    for (const Node& n : delta.added_nodes) {
        addNodeWithId(n.id, n.type);
        Node* node = getNode(n.id);
        node->attributes = n.attributes;
        node->activation = n.activation;
        node->uncertainty = n.uncertainty;
        node->metadata = n.metadata;
    }

    for (const Edge& e : delta.added_edges) {
        addEdgeWithId(e.id, e.source, e.target, e.relationship_type, e.weight);
    }

    for (const Node& n : delta.modified_nodes_after) {
        Node* node = getNode(n.id);
        if (node) {
            node->type = n.type;
            node->attributes = n.attributes;
            node->activation = n.activation;
            node->uncertainty = n.uncertainty;
            node->metadata = n.metadata;
        }
    }
    for (const Edge& e : delta.modified_edges_after) {
        Edge* edge = getEdge(e.id);
        if (edge) {
            edge->relationship_type = e.relationship_type;
            edge->weight = e.weight;
        }
    }
}

void Graph::undoDelta(const GraphDelta& delta) {
    for (const Node& n : delta.modified_nodes_before) {
        Node* node = getNode(n.id);
        if (node) {
            node->type = n.type;
            node->attributes = n.attributes;
            node->activation = n.activation;
            node->uncertainty = n.uncertainty;
            node->metadata = n.metadata;
        }
    }
    for (const Edge& e : delta.modified_edges_before) {
        Edge* edge = getEdge(e.id);
        if (edge) {
            edge->relationship_type = e.relationship_type;
            edge->weight = e.weight;
        }
    }

    for (const Edge& e : delta.added_edges) {
        removeEdge(e.id);
    }
    for (const Node& n : delta.added_nodes) {
        removeNode(n.id);
    }

    for (const Node& n : delta.removed_nodes) {
        if (!getNode(n.id)) {
            addNodeWithId(n.id, n.type);
        }
        Node* node = getNode(n.id);
        if (node) {
            node->attributes = n.attributes;
            node->activation = n.activation;
            node->uncertainty = n.uncertainty;
            node->metadata = n.metadata;
        }
    }
    for (const Edge& e : delta.removed_edges) {
        if (!getEdge(e.id) && getNode(e.source) && getNode(e.target)) {
            addEdgeWithId(e.id, e.source, e.target, e.relationship_type, e.weight);
        }
    }
}

// ─── Cloning ───────────────────────────────────────────────────

Graph Graph::clone() const {
    Graph copy;
    copy.next_node_id_ = next_node_id_;
    copy.next_edge_id_ = next_edge_id_;
    copy.nodes_ = nodes_;
    copy.edges_ = edges_;
    copy.outgoing_ = outgoing_;
    copy.incoming_ = incoming_;
    return copy;
}

// ─── Iteration ─────────────────────────────────────────────────

void Graph::forEachNode(std::function<void(const Node&)> fn) const {
    for (const auto& [_, node] : nodes_) {
        fn(node);
    }
}

void Graph::forEachEdge(std::function<void(const Edge&)> fn) const {
    for (const auto& [_, edge] : edges_) {
        fn(edge);
    }
}

} // namespace sare
