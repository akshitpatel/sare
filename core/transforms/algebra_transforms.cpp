#include "transforms/algebra_transforms.hpp"

namespace sare {

// ─── Additive Identity: x + 0 → x ─────────────────────────────

std::string AddZeroTransform::name() const { return "algebra_add_zero"; }

bool AddZeroTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type == "operator" && node.getAttribute("op") == "add") {
            auto outgoing = graph.getOutgoing(node.id);
            for (uint64_t eid : outgoing) {
                const Edge* e = graph.getEdge(eid);
                if (!e) continue;
                const Node* child = graph.getNode(e->target);
                if (child && child->type == "literal" && child->getAttribute("value") == "0") {
                    found = true;
                    return;
                }
            }
        }
    });
    return found;
}

GraphDelta AddZeroTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& node) {
        if (node.type != "operator" || node.getAttribute("op") != "add") return;

        auto outgoing = graph.getOutgoing(node.id);
        uint64_t zero_child_id = 0;
        uint64_t other_child_id = 0;
        bool has_zero = false;

        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            const Node* child = graph.getNode(e->target);
            if (!child) continue;

            if (child->type == "literal" && child->getAttribute("value") == "0") {
                zero_child_id = child->id;
                has_zero = true;
            } else {
                other_child_id = child->id;
            }
        }

        if (!has_zero || other_child_id == 0) return;

        delta.removed_node_ids.push_back(zero_child_id);
        delta.removed_node_ids.push_back(node.id);

        for (uint64_t eid : outgoing) {
            delta.removed_edge_ids.push_back(eid);
        }
        auto incoming = graph.getIncoming(node.id);
        for (uint64_t eid : incoming) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            delta.removed_edge_ids.push_back(eid);
            Edge new_edge;
            new_edge.id = eid + 10000;
            new_edge.source = e->source;
            new_edge.target = other_child_id;
            new_edge.relationship_type = e->relationship_type;
            new_edge.weight = e->weight;
            delta.added_edges.push_back(new_edge);
        }
    });

    return delta;
}

double AddZeroTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -1.0;
}

double AddZeroTransform::cost() const { return 0.1; }

// ─── Multiplicative Identity: x * 1 → x ───────────────────────

std::string MulOneTransform::name() const { return "algebra_mul_one"; }

bool MulOneTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type == "operator" && node.getAttribute("op") == "mul") {
            auto outgoing = graph.getOutgoing(node.id);
            for (uint64_t eid : outgoing) {
                const Edge* e = graph.getEdge(eid);
                if (!e) continue;
                const Node* child = graph.getNode(e->target);
                if (child && child->type == "literal" && child->getAttribute("value") == "1") {
                    found = true;
                    return;
                }
            }
        }
    });
    return found;
}

GraphDelta MulOneTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& node) {
        if (node.type != "operator" || node.getAttribute("op") != "mul") return;

        auto outgoing = graph.getOutgoing(node.id);
        uint64_t one_child_id = 0;
        uint64_t other_child_id = 0;
        bool has_one = false;

        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            const Node* child = graph.getNode(e->target);
            if (!child) continue;

            if (child->type == "literal" && child->getAttribute("value") == "1") {
                one_child_id = child->id;
                has_one = true;
            } else {
                other_child_id = child->id;
            }
        }

        if (!has_one || other_child_id == 0) return;

        delta.removed_node_ids.push_back(one_child_id);
        delta.removed_node_ids.push_back(node.id);

        for (uint64_t eid : outgoing) {
            delta.removed_edge_ids.push_back(eid);
        }
        auto incoming = graph.getIncoming(node.id);
        for (uint64_t eid : incoming) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            delta.removed_edge_ids.push_back(eid);
            Edge new_edge;
            new_edge.id = eid + 10000;
            new_edge.source = e->source;
            new_edge.target = other_child_id;
            new_edge.relationship_type = e->relationship_type;
            new_edge.weight = e->weight;
            delta.added_edges.push_back(new_edge);
        }
    });

    return delta;
}

double MulOneTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -1.0;
}

double MulOneTransform::cost() const { return 0.1; }

// ─── Multiplicative Zero: x * 0 → 0 ───────────────────────────

std::string MulZeroTransform::name() const { return "algebra_mul_zero"; }

bool MulZeroTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type == "operator" && node.getAttribute("op") == "mul") {
            auto outgoing = graph.getOutgoing(node.id);
            for (uint64_t eid : outgoing) {
                const Edge* e = graph.getEdge(eid);
                if (!e) continue;
                const Node* child = graph.getNode(e->target);
                if (child && child->type == "literal" && child->getAttribute("value") == "0") {
                    found = true;
                    return;
                }
            }
        }
    });
    return found;
}

GraphDelta MulZeroTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& node) {
        if (node.type != "operator" || node.getAttribute("op") != "mul") return;

        auto outgoing = graph.getOutgoing(node.id);
        bool has_zero = false;
        std::vector<uint64_t> children;

        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            const Node* child = graph.getNode(e->target);
            if (!child) continue;
            children.push_back(child->id);
            if (child->type == "literal" && child->getAttribute("value") == "0") {
                has_zero = true;
            }
        }

        if (!has_zero) return;

        Node modified = node;
        modified.type = "literal";
        modified.attributes.clear();
        modified.attributes["value"] = "0";
        modified.uncertainty = 0.0;

        delta.modified_nodes_before.push_back(node);
        delta.modified_nodes_after.push_back(modified);

        for (uint64_t eid : outgoing) {
            delta.removed_edge_ids.push_back(eid);
        }
        for (uint64_t cid : children) {
            delta.removed_node_ids.push_back(cid);
        }
    });

    return delta;
}

double MulZeroTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -2.0;
}

double MulZeroTransform::cost() const { return 0.1; }

} // namespace sare
