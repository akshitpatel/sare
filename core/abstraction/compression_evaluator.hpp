#pragma once

#include "abstraction/macro_builder.hpp"
#include "energy/energy.hpp"
#include "transforms/transform_registry.hpp"
#include <vector>

namespace sare {

// ─── Compression Evaluator ────────────────────────────────────
// Validates macro transforms against held-out problems before
// promoting them to the abstraction registry.
//
// A macro is valid if:
// 1. It produces equivalent energy reduction to its component steps
// 2. It achieves compression > γ (fewer search steps)
// 3. It generalizes across multiple problems

struct CompressionResult {
    std::string macro_name;
    double compression_ratio;    // C(P) = OrigSteps / CompressedSteps
    double energy_equivalence;   // 1.0 = exact match to component steps
    int problems_tested;
    int problems_improved;       // where macro helped
    bool recommended;            // meets all thresholds
};

class CompressionEvaluator {
public:
    CompressionEvaluator(double compression_threshold = 1.5,
                          double min_generalization = 0.5)
        : compression_threshold_(compression_threshold),
          min_generalization_(min_generalization) {}

    /// Evaluate a macro-transform against a set of test problems.
    CompressionResult evaluate(
        MacroTransform* macro,
        const std::vector<Graph>& test_problems,
        const EnergyAggregator& energy,
        const TransformRegistry& baseline);

private:
    double compression_threshold_;
    double min_generalization_;  // fraction of problems where macro must help
};

} // namespace sare
