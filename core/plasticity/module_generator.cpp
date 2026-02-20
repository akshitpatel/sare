#include "plasticity/module_generator.hpp"
#include "graph/graph_diff.hpp"
#include "transforms/algebra_transforms.hpp"
#include "transforms/logic_transforms.hpp"
#include "transforms/ast_transforms.hpp"
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace sare {

namespace {

std::unique_ptr<Transform> makePrimitive(const std::string& name) {
    if (name == "algebra_add_zero") return std::make_unique<AddZeroTransform>();
    if (name == "algebra_mul_one") return std::make_unique<MulOneTransform>();
    if (name == "algebra_mul_zero") return std::make_unique<MulZeroTransform>();
    if (name == "logic_double_negation") return std::make_unique<DoubleNegationTransform>();
    if (name == "logic_and_true") return std::make_unique<AndTrueTransform>();
    if (name == "ast_constant_fold") return std::make_unique<ConstantFoldTransform>();
    return nullptr;
}

bool isKnownPrimitive(const std::string& name) {
    return static_cast<bool>(makePrimitive(name));
}

std::string sanitizeToken(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    }
    return out;
}

class GeneratedMacroTransform final : public Transform {
public:
    GeneratedMacroTransform(std::string macro_name, std::vector<std::string> steps)
        : name_(std::move(macro_name)), steps_(std::move(steps)) {}

    std::string name() const override { return name_; }

    bool match(const Graph& graph) const override {
        if (steps_.empty()) return false;
        auto first = makePrimitive(steps_[0]);
        return first && first->match(graph);
    }

    GraphDelta apply(const Graph& graph) const override {
        Graph working = graph.clone();
        bool changed = false;

        for (const auto& step_name : steps_) {
            auto step = makePrimitive(step_name);
            if (!step || !step->match(working)) {
                break;
            }

            GraphDelta step_delta = step->apply(working);
            if (step_delta.empty()) {
                break;
            }
            working.applyDelta(step_delta);
            changed = true;
        }

        if (!changed) {
            return {};
        }
        return GraphDiff::diff(graph, working);
    }

    double estimateDeltaEnergy(const Graph& graph) const override {
        double estimate = 0.0;
        for (const auto& step_name : steps_) {
            auto step = makePrimitive(step_name);
            if (!step) continue;
            estimate += step->estimateDeltaEnergy(graph);
        }
        return estimate;
    }

    double cost() const override {
        double total = 0.0;
        for (const auto& step_name : steps_) {
            auto step = makePrimitive(step_name);
            if (!step) continue;
            total += step->cost();
        }
        return total;
    }

private:
    std::string name_;
    std::vector<std::string> steps_;
};

} // namespace

bool ModuleGenerator::isPersistentFailure(
    const std::vector<SolveEpisode>& episodes,
    double failure_rate_threshold) const {

    if (episodes.empty()) return false;

    int failures = 0;
    for (const auto& ep : episodes) {
        if (!ep.success) failures++;
    }

    double rate = static_cast<double>(failures) / episodes.size();
    return rate >= failure_rate_threshold;
}

std::vector<std::string> ModuleGenerator::extractFailureFeatures(
    const std::vector<SolveEpisode>& failures) const {

    // Count which transforms were attempted most often before failure
    std::unordered_map<std::string, int> transform_frequency;

    for (const auto& ep : failures) {
        if (ep.success) continue;
        for (const auto& t : ep.transform_sequence) {
            transform_frequency[t]++;
        }
    }

    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted(
        transform_frequency.begin(), transform_frequency.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top features
    std::vector<std::string> features;
    for (size_t i = 0; i < std::min(sorted.size(), size_t(5)); i++) {
        features.push_back(sorted[i].first);
    }
    return features;
}

std::vector<std::unique_ptr<Transform>> ModuleGenerator::generate(
    const std::vector<SolveEpisode>& failures,
    size_t max_candidates) {

    if (!isPersistentFailure(failures)) {
        return {};
    }

    std::vector<std::string> features = extractFailureFeatures(failures);
    std::vector<std::string> primitives;
    primitives.reserve(features.size());
    for (const auto& f : features) {
        if (isKnownPrimitive(f)) {
            primitives.push_back(f);
        }
    }

    if (primitives.size() < 2) {
        static const std::vector<std::string> fallback = {
            "algebra_add_zero",
            "algebra_mul_one",
            "ast_constant_fold",
        };
        for (const auto& f : fallback) {
            if (primitives.size() >= 2) break;
            if (std::find(primitives.begin(), primitives.end(), f) == primitives.end()) {
                primitives.push_back(f);
            }
        }
    }

    std::vector<std::unique_ptr<Transform>> candidates;
    if (primitives.size() < 2 || max_candidates == 0) {
        return candidates;
    }

    size_t created = 0;
    for (size_t i = 0; i < primitives.size() && created < max_candidates; i++) {
        for (size_t j = i + 1; j < primitives.size() && created < max_candidates; j++) {
            std::vector<std::string> steps = {primitives[i], primitives[j]};
            std::string macro_name =
                "generated_" + sanitizeToken(primitives[i]) + "_then_" + sanitizeToken(primitives[j]);
            candidates.push_back(std::make_unique<GeneratedMacroTransform>(macro_name, std::move(steps)));
            created++;
        }
    }

    return candidates;
}

} // namespace sare
