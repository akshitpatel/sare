#include <gtest/gtest.h>
#include "plasticity/module_generator.hpp"
#include "plasticity/pruning_manager.hpp"
#include "plasticity/sandbox_runner.hpp"
#include "energy/syntax_energy.hpp"

using namespace sare;

// ─── Module Generator Tests ───────────────────────────────────

TEST(PlasticityTest, PersistentFailureDetection) {
    ModuleGenerator gen;

    std::vector<SolveEpisode> episodes;
    // 4/5 failures = 80% failure rate
    for (int i = 0; i < 5; i++) {
        SolveEpisode ep;
        ep.problem_id = "prob_" + std::to_string(i);
        ep.success = (i == 0);  // only first succeeds
        ep.transform_sequence = {"add_zero", "mul_one"};
        episodes.push_back(ep);
    }

    EXPECT_TRUE(gen.isPersistentFailure(episodes, 0.5));
}

TEST(PlasticityTest, NoPersistentFailure) {
    ModuleGenerator gen;

    std::vector<SolveEpisode> episodes;
    for (int i = 0; i < 5; i++) {
        SolveEpisode ep;
        ep.success = true;
        episodes.push_back(ep);
    }

    EXPECT_FALSE(gen.isPersistentFailure(episodes, 0.5));
}

TEST(PlasticityTest, ExtractFailureFeatures) {
    ModuleGenerator gen;

    std::vector<SolveEpisode> failures;
    SolveEpisode ep1;
    ep1.success = false;
    ep1.transform_sequence = {"add_zero", "mul_one", "add_zero"};
    SolveEpisode ep2;
    ep2.success = false;
    ep2.transform_sequence = {"add_zero", "const_fold"};
    failures.push_back(ep1);
    failures.push_back(ep2);

    auto features = gen.extractFailureFeatures(failures);
    EXPECT_FALSE(features.empty());
    // "add_zero" appears 3 times across failures, should be top feature
    EXPECT_EQ(features[0], "add_zero");
}

TEST(PlasticityTest, GenerateCandidatesFromPersistentFailures) {
    ModuleGenerator gen;
    std::vector<SolveEpisode> episodes;

    for (int i = 0; i < 6; i++) {
        SolveEpisode ep;
        ep.success = false;
        ep.transform_sequence = {
            "algebra_add_zero",
            "algebra_mul_one",
            "ast_constant_fold",
        };
        episodes.push_back(ep);
    }

    auto generated = gen.generate(episodes, 3);
    EXPECT_FALSE(generated.empty());
}

// ─── Pruning Manager Tests ────────────────────────────────────

// Simple transform with controllable utility for testing
class TestPrunableTransform : public Transform {
public:
    TestPrunableTransform(const std::string& name, double util_bias)
        : name_(name), util_bias_(util_bias) {
        // Simulate some applications
        recordApplication(util_bias);
        recordApplication(util_bias);
    }
    std::string name() const override { return name_; }
    bool match(const Graph&) const override { return true; }
    GraphDelta apply(const Graph&) const override { return GraphDelta{}; }
    double estimateDeltaEnergy(const Graph&) const override { return util_bias_; }
    double cost() const override { return 0.0; }
private:
    std::string name_;
    double util_bias_;
};

TEST(PlasticityTest, PruningIdentifiesCandidates) {
    TransformRegistry registry;

    // Good transform (positive utility)
    registry.registerTransform(
        std::make_unique<TestPrunableTransform>("good_transform", -5.0));

    // Bad transform (negative utility → positive delta_e)
    registry.registerTransform(
        std::make_unique<TestPrunableTransform>("bad_transform", 5.0));

    PruningManager pm(0.0);
    auto candidates = pm.identifyCandidates(registry);

    // "bad_transform" has negative utility (positive ΔE = bad)
    EXPECT_EQ(candidates.size(), 1);
    EXPECT_EQ(candidates[0], "bad_transform");
}

TEST(PlasticityTest, PruningRemovesUnderperformers) {
    TransformRegistry registry;
    registry.registerTransform(
        std::make_unique<TestPrunableTransform>("good", -5.0));
    registry.registerTransform(
        std::make_unique<TestPrunableTransform>("bad", 5.0));

    EXPECT_EQ(registry.count(), 2);

    PruningManager pm(0.0);
    int removed = pm.pruneUnderperformers(registry);

    EXPECT_EQ(removed, 1);
    EXPECT_EQ(registry.count(), 1);
    EXPECT_NE(registry.getByName("good"), nullptr);
    EXPECT_EQ(registry.getByName("bad"), nullptr);
}

class FixErrorNodeTransform : public Transform {
public:
    std::string name() const override { return "fix_error_node"; }
    bool match(const Graph& graph) const override {
        bool found = false;
        graph.forEachNode([&](const Node& node) {
            if (node.type == "error") {
                found = true;
            }
        });
        return found;
    }
    GraphDelta apply(const Graph& graph) const override {
        GraphDelta delta;
        graph.forEachNode([&](const Node& node) {
            if (node.type == "error") {
                delta.removed_node_ids.push_back(node.id);
            }
        });
        return delta;
    }
    double estimateDeltaEnergy(const Graph&) const override { return -1.0; }
    double cost() const override { return 0.1; }
};

TEST(PlasticityTest, SandboxUsesBaselineVsCandidateSearch) {
    Graph problem;
    problem.addNode("error");
    std::vector<Graph> problems = {problem};

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<SyntaxEnergy>());

    TransformRegistry baseline;
    FixErrorNodeTransform candidate;

    SearchConfig cfg;
    cfg.beam_width = 2;
    cfg.max_depth = 3;
    cfg.budget_seconds = 1.0;
    cfg.max_expansions = 20;

    SandboxRunner runner(0.0);
    SandboxResult result = runner.evaluate(&candidate, problems, energy, baseline, cfg);

    EXPECT_EQ(result.problems_tested, 1);
    EXPECT_EQ(result.problems_improved, 1);
    EXPECT_GT(result.performance_delta, 0.0);
    EXPECT_TRUE(result.promoted);
    EXPECT_EQ(baseline.getByName(candidate.name()), nullptr);
}
