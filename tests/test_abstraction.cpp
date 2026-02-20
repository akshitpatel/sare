#include <gtest/gtest.h>
#include "abstraction/trace_miner.hpp"
#include "abstraction/macro_builder.hpp"
#include "abstraction/abstraction_registry.hpp"
#include "abstraction/compression_evaluator.hpp"

using namespace sare;

// ─── Trace Miner Tests ────────────────────────────────────────

TEST(AbstractionTest, TraceMinerBasic) {
    TraceMiner miner;
    EXPECT_EQ(miner.traceCount(), 0);

    miner.addTrace({"add_zero", "mul_one", "const_fold"});
    miner.addTrace({"add_zero", "mul_one", "simplify"});
    miner.addTrace({"add_zero", "mul_one", "const_fold"});
    EXPECT_EQ(miner.traceCount(), 3);
}

TEST(AbstractionTest, TraceMinerFindsFrequentPatterns) {
    TraceMiner miner;

    // The pattern "add_zero→mul_one" appears in 3/4 traces
    miner.addTrace({"add_zero", "mul_one", "const_fold"});
    miner.addTrace({"add_zero", "mul_one", "simplify"});
    miner.addTrace({"add_zero", "mul_one", "const_fold"});
    miner.addTrace({"const_fold", "simplify"});

    auto patterns = miner.mine(2, 2, 3);  // min_freq=2, length=2..3
    EXPECT_GT(patterns.size(), 0);

    // The most frequent 2-gram should be add_zero→mul_one
    bool found = false;
    for (const auto& p : patterns) {
        if (p.transform_subsequence.size() == 2 &&
            p.transform_subsequence[0] == "add_zero" &&
            p.transform_subsequence[1] == "mul_one") {
            found = true;
            EXPECT_GE(p.frequency, 3);
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(AbstractionTest, TraceMinerCompressionFilter) {
    TraceMiner miner;
    miner.addTrace({"a", "b", "c"});
    miner.addTrace({"a", "b", "c"});
    miner.addTrace({"a", "b", "d"});

    // 3-length patterns should have compression = 3.0
    auto patterns = miner.mineWithCompression(2, 2.5, 3, 3);
    // Only patterns with length >= 3 meet compression > 2.5
    for (const auto& p : patterns) {
        EXPECT_GE(p.avg_compression_ratio, 2.5);
    }
}

// ─── MacroTransform Tests ──────────────────────────────────────

TEST(AbstractionTest, MacroBuilderGeneratesName) {
    TransformPattern pattern;
    pattern.transform_subsequence = {"add_zero", "mul_one"};
    pattern.frequency = 5;

    std::string name = MacroBuilder::generateName(pattern);
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name.find("macro_"), std::string::npos);
}

// ─── Abstraction Registry Tests ────────────────────────────────

TEST(AbstractionTest, RegistryPromoteAndLookup) {
    AbstractionRegistry registry;
    EXPECT_EQ(registry.count(), 0);

    auto macro = std::make_unique<MacroTransform>(
        "test_macro", std::vector<std::string>{"add_zero", "mul_one"});

    registry.promote(std::move(macro), "algebra");
    EXPECT_EQ(registry.count(), 1);

    auto found = registry.getByName("test_macro");
    EXPECT_NE(found, nullptr);
    EXPECT_EQ(found->stepCount(), 2);
}

TEST(AbstractionTest, RegistryDemote) {
    AbstractionRegistry registry;
    auto macro = std::make_unique<MacroTransform>(
        "to_remove", std::vector<std::string>{"a", "b"});
    registry.promote(std::move(macro), "test");
    EXPECT_EQ(registry.count(), 1);

    bool removed = registry.demote("to_remove");
    EXPECT_TRUE(removed);
    EXPECT_EQ(registry.count(), 0);
}

TEST(AbstractionTest, RegistryDomainLookup) {
    AbstractionRegistry registry;
    registry.promote(std::make_unique<MacroTransform>(
        "alg1", std::vector<std::string>{"a"}), "algebra");
    registry.promote(std::make_unique<MacroTransform>(
        "log1", std::vector<std::string>{"b"}), "logic");

    auto algebra = registry.getForDomain("algebra");
    EXPECT_EQ(algebra.size(), 1);
    EXPECT_EQ(algebra[0]->name(), "alg1");

    auto logic = registry.getForDomain("logic");
    EXPECT_EQ(logic.size(), 1);
}
