#include "transforms/logic_transforms.hpp"

namespace sare {

// ─── Double Negation Elimination: not(not(x)) → x ─────────────

std::string DoubleNegationTransform::name() const { return "logic_double_negation"; }

bool DoubleNegationTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type == "operator" && node.getAttribute("op") == "not") {
            auto outgoing = graph.getOutgoing(node.id);
            for (uint64_t eid : outgoing) {
                const Edge* e = graph.getEdge(eid);
                if (!e) continue;
                const Node* child = graph.getNode(e->target);
                if (child && child->type == "operator" && child->getAttribute("op") == "not") {
                    found = true;
                    return;
                }
            }
        }
    });
    return found;
}

GraphDelta DoubleNegationTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& outer) {
        if (outer.type != "operator" || outer.getAttribute("op") != "not") return;

        auto outer_out = graph.getOutgoing(outer.id);
        for (uint64_t eid : outer_out) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            const Node* inner = graph.getNode(e->target);
            if (!inner || inner->type != "operator" || inner->getAttribute("op") != "not") continue;

            auto inner_out = graph.getOutgoing(inner->id);
            if (inner_out.empty()) continue;
            const Edge* inner_edge = graph.getEdge(inner_out[0]);
            if (!inner_edge) continue;
            uint64_t actual_child = inner_edge->target;

            delta.removed_node_ids.push_back(outer.id);
            delta.removed_node_ids.push_back(inner->id);

            delta.removed_edge_ids.push_back(eid);
            delta.removed_edge_ids.push_back(inner_out[0]);

            auto incoming = graph.getIncoming(outer.id);
            for (uint64_t pid : incoming) {
                const Edge* pe = graph.getEdge(pid);
                if (!pe) continue;
                delta.removed_edge_ids.push_back(pid);
                Edge new_edge;
                new_edge.id = pid + 10000;
                new_edge.source = pe->source;
                new_edge.target = actual_child;
                new_edge.relationship_type = pe->relationship_type;
                new_edge.weight = pe->weight;
                delta.added_edges.push_back(new_edge);
            }

            return;
        }
    });

    return delta;
}

double DoubleNegationTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -1.5;
}

double DoubleNegationTransform::cost() const { return 0.1; }

// ─── Identity: x AND true → x ─────────────────────────────────

std::string AndTrueTransform::name() const { return "logic_and_true"; }

bool AndTrueTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type == "operator" && node.getAttribute("op") == "and") {
            auto outgoing = graph.getOutgoing(node.id);
            for (uint64_t eid : outgoing) {
                const Edge* e = graph.getEdge(eid);
                if (!e) continue;
                const Node* child = graph.getNode(e->target);
                if (child && child->type == "literal" && child->getAttribute("value") == "true") {
                    found = true;
                    return;
                }
            }
        }
    });
    return found;
}

GraphDelta AndTrueTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& node) {
        if (node.type != "operator" || node.getAttribute("op") != "and") return;

        auto outgoing = graph.getOutgoing(node.id);
        uint64_t true_child_id = 0;
        uint64_t other_child_id = 0;
        bool has_true = false;

        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) continue;
            const Node* child = graph.getNode(e->target);
            if (!child) continue;

            if (child->type == "literal" && child->getAttribute("value") == "true") {
                true_child_id = child->id;
                has_true = true;
            } else {
                other_child_id = child->id;
            }
        }

        if (!has_true || other_child_id == 0) return;

        delta.removed_node_ids.push_back(true_child_id);
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

double AndTrueTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -1.0;
}

double AndTrueTransform::cost() const { return 0.1; }

} // namespace sare
