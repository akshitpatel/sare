#include "reflection_engine.hpp"
#include <algorithm>
#include <sstream>
#include <map>

namespace sare {

// ─── Public: reflect ──────────────────────────────────────────

std::unique_ptr<AbstractRule> ReflectionEngine::reflect(
    const Graph& initial, const Graph& final_g)
{
    if (initial.nodeCount() == 0 || final_g.nodeCount() == 0)
        return nullptr;

    ReflectionDiff diff = computeDiff(initial, final_g);
    if (diff.empty())
        return nullptr;

    return generalize(initial, final_g, diff);
}

// ─── Public: computeDiff ──────────────────────────────────────

ReflectionDiff ReflectionEngine::computeDiff(
    const Graph& initial, const Graph& final_g) const
{
    ReflectionDiff diff;

    // Added: in final but not initial
    final_g.forEachNode([&](const Node& node) {
        if (initial.getNode(node.id) == nullptr)
            diff.added_nodes.insert(node.id);
    });

    // Removed: in initial but not final
    initial.forEachNode([&](const Node& node) {
        if (final_g.getNode(node.id) == nullptr)
            diff.removed_nodes.insert(node.id);
    });

    // Modified: type, attributes, or edge cardinality changed
    initial.forEachNode([&](const Node& node_i) {
        const Node* node_f = final_g.getNode(node_i.id);
        if (!node_f) return;

        bool changed = (node_i.type != node_f->type ||
                        node_i.attributes != node_f->attributes);
        if (!changed) {
            auto out_i = initial.getOutgoing(node_i.id);
            auto out_f = final_g.getOutgoing(node_i.id);
            if (out_i.size() != out_f.size()) changed = true;
        }
        if (changed) diff.modified_nodes.insert(node_i.id);
    });

    // Context: unchanged neighbors of changed nodes
    std::unordered_set<uint64_t> all_changed;
    all_changed.insert(diff.removed_nodes.begin(), diff.removed_nodes.end());
    all_changed.insert(diff.added_nodes.begin(), diff.added_nodes.end());
    all_changed.insert(diff.modified_nodes.begin(), diff.modified_nodes.end());

    auto collect_context = [&](const Graph& g) {
        for (uint64_t id : all_changed) {
            if (!g.getNode(id)) continue;
            for (uint64_t nid : g.getNeighborNodes(id)) {
                if (all_changed.find(nid) == all_changed.end())
                    diff.context_nodes.insert(nid);
            }
        }
    };
    collect_context(initial);
    collect_context(final_g);

    return diff;
}

// ─── Public: generalize ───────────────────────────────────────

std::unique_ptr<AbstractRule> ReflectionEngine::generalize(
    const Graph& initial,
    const Graph& final_g,
    const ReflectionDiff& diff) const
{
    auto rule = std::make_unique<AbstractRule>();

    // Pattern = removed + modified + context (from initial)
    std::unordered_set<uint64_t> pattern_ids;
    pattern_ids.insert(diff.removed_nodes.begin(), diff.removed_nodes.end());
    pattern_ids.insert(diff.modified_nodes.begin(), diff.modified_nodes.end());
    pattern_ids.insert(diff.context_nodes.begin(), diff.context_nodes.end());

    // Replacement = added + modified + context (from final)
    std::unordered_set<uint64_t> replacement_ids;
    replacement_ids.insert(diff.added_nodes.begin(), diff.added_nodes.end());
    replacement_ids.insert(diff.modified_nodes.begin(), diff.modified_nodes.end());
    replacement_ids.insert(diff.context_nodes.begin(), diff.context_nodes.end());

    rule->pattern     = initial.extractSubgraph(pattern_ids);
    rule->replacement = final_g.extractSubgraph(replacement_ids);

    if (rule->pattern.nodeCount() == 0)
        return nullptr;

    rule->name              = inferRuleName(initial, diff);
    rule->domain            = inferDomain(rule->pattern);
    rule->type_constraints  = extractTypeConstraints(initial, pattern_ids);

    return rule;
}

// ─── Private: inferRuleName ───────────────────────────────────

std::string ReflectionEngine::inferRuleName(
    const Graph& initial, const ReflectionDiff& diff) const
{
    // Strategy: analyse the removed nodes to find the dominant pattern.
    //
    // Heuristic rules (ordered by specificity):
    //   "op(+) + child(constant:0)"  → "additive_identity"
    //   "op(*) + child(constant:1)"  → "multiplicative_identity"
    //   "op(*) + child(constant:0)"  → "multiplicative_zero"
    //   "op(neg) → op(neg) → X"     → "double_negation"
    //   const op const              → "constant_folding"
    //   fallback                    → "structural_reduction_<hash>"

    // Collect removed operator nodes
    for (uint64_t rid : diff.removed_nodes) {
        const Node* n = initial.getNode(rid);
        if (!n || n->type != "operator") continue;

        std::string op_label;
        auto it = n->attributes.find("label");
        if (it != n->attributes.end()) op_label = it->second;

        auto children = initial.getOutgoing(rid);
        bool has_zero = false, has_one = false, has_neg_child = false;
        bool children_all_const = !children.empty();

        for (auto cid : children) {
            const Node* c = initial.getNode(cid);
            if (!c) { children_all_const = false; continue; }
            std::string clabel;
            auto lit = c->attributes.find("label");
            if (lit != c->attributes.end()) clabel = lit->second;
            if (c->type == "constant" && clabel == "0") has_zero = true;
            if (c->type == "constant" && clabel == "1") has_one  = true;
            if (c->type == "operator" ) {
                std::string cl;
                auto clit = c->attributes.find("label");
                if (clit != c->attributes.end()) cl = clit->second;
                if (cl == "neg" || cl == "-") has_neg_child = true;
            }
            if (c->type != "constant") children_all_const = false;
        }

        if ((op_label == "+" || op_label == "add") && has_zero)
            return "additive_identity";
        if ((op_label == "*" || op_label == "mul") && has_one)
            return "multiplicative_identity";
        if ((op_label == "*" || op_label == "mul") && has_zero)
            return "multiplicative_zero";
        if ((op_label == "neg" || op_label == "-") && has_neg_child)
            return "double_negation";
        if (children_all_const && children.size() >= 2)
            return "constant_folding";
    }

    // Fallback: encode the diff size as a hash-like suffix
    std::ostringstream oss;
    oss << "structural_reduction_"
        << diff.removed_nodes.size() << "r"
        << diff.added_nodes.size() << "a";
    return oss.str();
}

// ─── Private: inferDomain ─────────────────────────────────────

std::string ReflectionEngine::inferDomain(const Graph& pattern) const
{
    // Heuristic: inspect operator labels to classify domain.
    bool has_arithmetic = false, has_logic = false, has_set = false;

    pattern.forEachNode([&](const Node& n) {
        if (n.type != "operator") return;
        std::string lbl;
        auto it = n.attributes.find("label");
        if (it != n.attributes.end()) lbl = it->second;

        if (lbl == "+" || lbl == "-" || lbl == "*" || lbl == "/" ||
            lbl == "add" || lbl == "sub" || lbl == "mul" || lbl == "div" ||
            lbl == "neg")
            has_arithmetic = true;
        else if (lbl == "and" || lbl == "or" || lbl == "not" ||
                 lbl == "implies" || lbl == "xor")
            has_logic = true;
        else if (lbl == "union" || lbl == "intersect" || lbl == "complement")
            has_set = true;
    });

    if (has_logic)     return "logic";
    if (has_set)       return "set_theory";
    if (has_arithmetic) return "arithmetic";
    return "general";
}

// ─── Private: extractTypeConstraints ─────────────────────────

std::vector<TypeConstraint> ReflectionEngine::extractTypeConstraints(
    const Graph& initial,
    const std::unordered_set<uint64_t>& pattern_ids) const
{
    std::vector<TypeConstraint> constraints;

    for (uint64_t nid : pattern_ids) {
        const Node* n = initial.getNode(nid);
        if (!n) continue;

        TypeConstraint tc;
        tc.node_id       = nid;
        tc.required_type = n->type;

        // Include the label only for constants/operators (semantically fixed).
        // For variables, leave required_label empty (matches any variable).
        if (n->type == "constant" || n->type == "operator") {
            auto it = n->attributes.find("label");
            if (it != n->attributes.end())
                tc.required_label = it->second;
        }
        constraints.push_back(tc);
    }

    return constraints;
}

} // namespace sare
