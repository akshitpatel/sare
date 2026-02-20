#include "abstraction/macro_builder.hpp"
#include <sstream>

namespace sare {

// ─── MacroTransform ───────────────────────────────────────────

bool MacroTransform::match(const Graph& graph) const {
    if (steps_.empty()) return false;
    // First step must match
    return steps_[0]->match(graph);
}

GraphDelta MacroTransform::apply(const Graph& graph) const {
    if (steps_.empty()) return GraphDelta{};

    // Apply all steps sequentially on a working copy
    Graph working = graph.clone();
    GraphDelta combined;

    for (Transform* step : steps_) {
        if (!step->match(working)) break;

        GraphDelta step_delta = step->apply(working);
        if (step_delta.empty()) break;

        working.applyDelta(step_delta);

        // Merge step_delta into combined delta
        for (auto& n : step_delta.added_nodes)
            combined.added_nodes.push_back(n);
        for (auto& e : step_delta.added_edges)
            combined.added_edges.push_back(e);
        for (auto& id : step_delta.removed_node_ids)
            combined.removed_node_ids.push_back(id);
        for (auto& id : step_delta.removed_edge_ids)
            combined.removed_edge_ids.push_back(id);
        for (auto& n : step_delta.modified_nodes_before)
            combined.modified_nodes_before.push_back(n);
        for (auto& n : step_delta.modified_nodes_after)
            combined.modified_nodes_after.push_back(n);
    }

    return combined;
}

double MacroTransform::estimateDeltaEnergy(const Graph& graph) const {
    double total = 0.0;
    for (Transform* step : steps_) {
        total += step->estimateDeltaEnergy(graph);
    }
    return total;
}

double MacroTransform::cost() const {
    double total = 0.0;
    for (Transform* step : steps_) {
        total += step->cost();
    }
    return total;
}

void MacroTransform::bindSteps(const TransformRegistry& registry) {
    steps_.clear();
    for (const auto& name : step_names_) {
        Transform* t = registry.getByName(name);
        if (t) steps_.push_back(t);
    }
}

// ─── MacroBuilder ─────────────────────────────────────────────

std::unique_ptr<MacroTransform> MacroBuilder::build(
    const TransformPattern& pattern,
    const TransformRegistry& registry) {

    std::string name = generateName(pattern);
    auto macro = std::make_unique<MacroTransform>(name,
                                                    pattern.transform_subsequence);
    macro->bindSteps(registry);
    return macro;
}

std::string MacroBuilder::generateName(const TransformPattern& pattern) {
    std::ostringstream oss;
    oss << "macro_";
    for (size_t i = 0; i < pattern.transform_subsequence.size(); i++) {
        if (i > 0) oss << "+";
        // Abbreviate: take first 4 chars of each name
        const auto& name = pattern.transform_subsequence[i];
        oss << name.substr(0, std::min(name.size(), size_t(4)));
    }
    return oss.str();
}

} // namespace sare
