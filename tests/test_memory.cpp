#include <gtest/gtest.h>
#include "memory/episodic_store.hpp"
#include "memory/strategy_memory.hpp"
#include "memory/graph_signature.hpp"

using namespace sare;

// ─── Episodic Store Tests ──────────────────────────────────────

TEST(MemoryTest, EpisodicStoreBasic) {
    EpisodicStore store;
    EXPECT_EQ(store.count(), 0);

    SolveEpisode ep;
    ep.problem_id = "test_problem_1";
    ep.transform_sequence = {"add_zero", "mul_one"};
    ep.energy_trajectory = {10.0, 5.0, 1.0};
    ep.initial_energy = 10.0;
    ep.final_energy = 1.0;
    ep.success = true;

    store.store(ep);
    EXPECT_EQ(store.count(), 1);
}

TEST(MemoryTest, EpisodicStoreRetrieve) {
    EpisodicStore store;

    for (int i = 0; i < 5; i++) {
        SolveEpisode ep;
        ep.problem_id = "prob_" + std::to_string(i);
        ep.success = (i % 2 == 0);
        ep.initial_energy = 10.0;
        ep.final_energy = ep.success ? 0.5 : 8.0;
        store.store(ep);
    }

    EXPECT_EQ(store.count(), 5);

    auto successes = store.retrieve(0, true);
    EXPECT_EQ(successes.size(), 3);  // indices 0, 2, 4

    auto recent = store.retrieveRecent(2);
    EXPECT_EQ(recent.size(), 2);
}

TEST(MemoryTest, EpisodicStoreSuccessRate) {
    EpisodicStore store;

    SolveEpisode success;
    success.success = true;
    success.initial_energy = 10.0;
    success.final_energy = 0.5;

    SolveEpisode failure;
    failure.success = false;
    failure.initial_energy = 10.0;
    failure.final_energy = 8.0;

    store.store(success);
    store.store(success);
    store.store(failure);

    EXPECT_NEAR(store.successRate(), 2.0/3.0, 0.01);
    EXPECT_NEAR(store.avgEnergyReduction(), 9.5, 0.01);
}

// ─── Strategy Memory Tests ─────────────────────────────────────

TEST(MemoryTest, StrategyMemoryBasic) {
    StrategyMemory mem;
    EXPECT_EQ(mem.count(), 0);

    Strategy s;
    s.signature = "N3E2|var:2,op:1|";
    s.transform_sequence = {"add_zero", "mul_one"};
    s.avg_energy_reduction = 5.0;
    s.success_rate = 0.9;

    mem.record("N3E2|var:2,op:1|", s);
    EXPECT_EQ(mem.count(), 1);

    auto result = mem.lookup("N3E2|var:2,op:1|");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->transform_sequence.size(), 2);
}

TEST(MemoryTest, StrategyMemoryLookupMiss) {
    StrategyMemory mem;
    auto result = mem.lookup("nonexistent");
    EXPECT_FALSE(result.has_value());
}

TEST(MemoryTest, StrategyMemoryDecay) {
    StrategyMemory mem;

    Strategy s;
    s.avg_energy_reduction = 0.001;
    s.success_rate = 0.005;

    mem.record("weak_sig", s);
    EXPECT_EQ(mem.count(), 1);

    // Decay should remove very weak strategies
    mem.decay(0.1);
    EXPECT_EQ(mem.count(), 0);
}

// ─── Graph Signature Tests ─────────────────────────────────────

TEST(MemoryTest, GraphSignatureCompute) {
    Graph g;
    g.addNode("variable");
    g.addNode("operator");
    g.addNode("variable");

    std::string sig = GraphSignature::compute(g);
    EXPECT_FALSE(sig.empty());
    EXPECT_NE(sig.find("N3"), std::string::npos);
}

TEST(MemoryTest, GraphSignatureEqualGraphs) {
    Graph g1, g2;
    g1.addNode("variable");
    g1.addNode("operator");
    g2.addNode("variable");
    g2.addNode("operator");

    std::string sig1 = GraphSignature::compute(g1);
    std::string sig2 = GraphSignature::compute(g2);

    double sim = GraphSignature::similarity(sig1, sig2);
    EXPECT_NEAR(sim, 1.0, 0.01);
}

TEST(MemoryTest, GraphSignatureDifferentGraphs) {
    Graph g1, g2;
    g1.addNode("variable");
    g1.addNode("variable");
    g2.addNode("operator");
    g2.addNode("operator");
    g2.addNode("operator");

    std::string sig1 = GraphSignature::compute(g1);
    std::string sig2 = GraphSignature::compute(g2);

    double sim = GraphSignature::similarity(sig1, sig2);
    EXPECT_LT(sim, 1.0);
}
