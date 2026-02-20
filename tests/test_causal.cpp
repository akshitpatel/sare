#include <gtest/gtest.h>
#include "causal/intervention.hpp"
#include "causal/counterfactual.hpp"
#include "causal/hypothesis_ranker.hpp"

using namespace sare;

// Simple energy component for causal tests
class CausalTestEnergy : public EnergyComponent {
public:
    double compute(const Graph& g) const override {
        double total = 0.0;
        g.forEachNode([&](const Node& n) {
            total += computeNode(g, n.id);
        });
        return total;
    }

    double computeNode(const Graph& g, uint64_t node_id) const override {
        const Node* n = g.getNode(node_id);
        if (!n) return 0.0;
        // Energy based on uncertainty
        return n->uncertainty * 2.0;
    }

    std::string name() const override { return "uncertainty"; }
};

// ─── Intervention Tests ────────────────────────────────────────

TEST(CausalTest, DoIntervention) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    Node* node = g.getNode(n1);
    node->attributes["value"] = "unknown";
    node->uncertainty = 1.0;

    InterventionEngine engine;
    Intervention i;
    i.node_id = n1;
    i.attribute = "value";
    i.value = "42";

    Graph forked = engine.doIntervention(g, i);

    // Original unchanged
    EXPECT_EQ(g.getNode(n1)->attributes["value"], "unknown");
    EXPECT_NEAR(g.getNode(n1)->uncertainty, 1.0, 0.001);

    // Forked has intervention applied
    EXPECT_EQ(forked.getNode(n1)->attributes["value"], "42");
    EXPECT_NEAR(forked.getNode(n1)->uncertainty, 0.0, 0.001);
}

TEST(CausalTest, CompareEnergies) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    g.getNode(n1)->uncertainty = 1.0;

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<CausalTestEnergy>());

    InterventionEngine engine;
    Intervention i;
    i.node_id = n1;
    i.attribute = "value";
    i.value = "known";

    Graph forked = engine.doIntervention(g, i);

    double delta = engine.compareEnergies(g, forked, energy);
    // Intervention reduces uncertainty → lower energy
    EXPECT_LT(delta, 0.0);
}

TEST(CausalTest, HasCausalEffect) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    g.getNode(n1)->uncertainty = 1.0;

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<CausalTestEnergy>());

    InterventionEngine engine;
    Intervention i;
    i.node_id = n1;
    i.attribute = "value";
    i.value = "42";

    EXPECT_TRUE(engine.hasCausalEffect(g, i, energy, 0.01));
}

// ─── Counterfactual Tests ──────────────────────────────────────

TEST(CausalTest, CounterfactualSimulate) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    uint64_t n2 = g.addNode("variable");
    g.getNode(n1)->uncertainty = 1.0;
    g.getNode(n2)->uncertainty = 0.5;

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<CausalTestEnergy>());

    CounterfactualSimulator sim;
    Intervention i;
    i.node_id = n1;
    i.attribute = "value";
    i.value = "known";

    auto result = sim.simulate(g, i, energy);
    EXPECT_LT(result.delta, 0.0);  // intervention reduces energy
}

TEST(CausalTest, CompareMultipleInterventions) {
    Graph g;
    uint64_t n1 = g.addNode("variable");
    uint64_t n2 = g.addNode("variable");
    g.getNode(n1)->uncertainty = 1.0;    // high uncertainty
    g.getNode(n2)->uncertainty = 0.1;    // low uncertainty

    EnergyAggregator energy;
    energy.addComponent(std::make_unique<CausalTestEnergy>());

    CounterfactualSimulator sim;

    Intervention i1{n1, "value", "known"};
    Intervention i2{n2, "value", "known"};

    auto results = sim.compareInterventions(g, {i1, i2}, energy);
    EXPECT_EQ(results.size(), 2);

    // First result should be most impactful (n1 has higher uncertainty)
    EXPECT_GT(std::abs(results[0].delta), std::abs(results[1].delta));
}

// ─── Hypothesis Ranker Tests ───────────────────────────────────

TEST(CausalTest, HypothesisRanking) {
    HypothesisRanker ranker(1.0);  // λ = 1

    std::vector<CausalHypothesis> hypotheses;

    CausalHypothesis h1;
    h1.name = "simple";
    h1.prediction_error = 0.5;
    h1.complexity = 1.0;

    CausalHypothesis h2;
    h2.name = "complex";
    h2.prediction_error = 0.1;
    h2.complexity = 5.0;

    CausalHypothesis h3;
    h3.name = "balanced";
    h3.prediction_error = 0.3;
    h3.complexity = 1.5;

    hypotheses = {h1, h2, h3};
    auto ranked = ranker.rank(hypotheses);

    // Scores: simple=1.5, complex=5.1, balanced=1.8
    // Best should be "simple" (lowest score)
    EXPECT_EQ(ranked[0].name, "simple");
    EXPECT_LT(ranked[0].score, ranked[1].score);
}

TEST(CausalTest, HypothesisLambdaEffect) {
    // With low lambda, complexity matters less → complex model might win
    HypothesisRanker ranker(0.01);

    std::vector<CausalHypothesis> hypotheses;

    CausalHypothesis simple;
    simple.name = "simple";
    simple.prediction_error = 0.5;
    simple.complexity = 1.0;

    CausalHypothesis complex;
    complex.name = "complex";
    complex.prediction_error = 0.1;
    complex.complexity = 5.0;

    hypotheses = {simple, complex};
    auto best = ranker.best(hypotheses);

    // With λ=0.01: simple=0.51, complex=0.15 → complex wins
    EXPECT_EQ(best.name, "complex");
}
