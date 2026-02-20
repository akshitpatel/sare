#include "transforms/ast_transforms.hpp"

namespace sare {

// ─── Constant Folding: literal op literal → result ─────────────
// Evaluates binary operations on two literal operands.

std::string ConstantFoldTransform::name() const { return "ast_constant_fold"; }

bool ConstantFoldTransform::match(const Graph& graph) const {
    bool found = false;
    graph.forEachNode([&](const Node& node) {
        if (found) return;
        if (node.type != "operator") return;

        auto outgoing = graph.getOutgoing(node.id);
        if (outgoing.size() != 2) return;

        bool all_literal = true;
        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) {
                all_literal = false;
                break;
            }
            const Node* child = graph.getNode(e->target);
            if (!child || child->type != "literal" || !child->hasAttribute("value")) {
                all_literal = false;
                break;
            }
        }
        if (all_literal) found = true;
    });
    return found;
}

GraphDelta ConstantFoldTransform::apply(const Graph& graph) const {
    GraphDelta delta;

    graph.forEachNode([&](const Node& node) {
        if (node.type != "operator") return;
        std::string op = node.getAttribute("op");
        if (op.empty()) return;

        auto outgoing = graph.getOutgoing(node.id);
        if (outgoing.size() != 2) return;

        std::vector<std::pair<uint64_t, double>> children;
        bool all_numeric = true;
        for (uint64_t eid : outgoing) {
            const Edge* e = graph.getEdge(eid);
            if (!e) {
                all_numeric = false;
                break;
            }
            const Node* child = graph.getNode(e->target);
            if (!child || child->type != "literal" || !child->hasAttribute("value")) {
                all_numeric = false;
                break;
            }
            try {
                double val = std::stod(child->getAttribute("value"));
                children.emplace_back(child->id, val);
            } catch (...) {
                all_numeric = false;
                break;
            }
        }
        if (!all_numeric || children.size() != 2) return;

        double result = 0.0;
        if (op == "add") result = children[0].second + children[1].second;
        else if (op == "sub") result = children[0].second - children[1].second;
        else if (op == "mul") result = children[0].second * children[1].second;
        else if (op == "div" && children[1].second != 0) result = children[0].second / children[1].second;
        else return;

        Node modified = node;
        modified.type = "literal";
        modified.attributes.clear();
        modified.attributes["value"] = std::to_string(result);
        modified.uncertainty = 0.0;

        delta.modified_nodes_before.push_back(node);
        delta.modified_nodes_after.push_back(modified);

        for (uint64_t eid : outgoing) {
            delta.removed_edge_ids.push_back(eid);
        }
        for (const auto& [cid, _] : children) {
            delta.removed_node_ids.push_back(cid);
        }
    });

    return delta;
}

double ConstantFoldTransform::estimateDeltaEnergy(const Graph& /*graph*/) const {
    return -2.0;
}

double ConstantFoldTransform::cost() const { return 0.2; }

} // namespace sare
