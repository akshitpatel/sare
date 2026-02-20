#include "abstraction/compression_evaluator.hpp"

namespace sare {

CompressionResult CompressionEvaluator::evaluate(
    MacroTransform* macro,
    const std::vector<Graph>& test_problems,
    const EnergyAggregator& energy,
    const TransformRegistry& baseline) {

    CompressionResult result;
    result.macro_name = macro->name();
    result.compression_ratio = static_cast<double>(macro->stepCount());
    result.problems_tested = static_cast<int>(test_problems.size());
    result.problems_improved = 0;
    result.energy_equivalence = 0.0;

    if (test_problems.empty()) {
        result.recommended = false;
        return result;
    }

    double total_equivalence = 0.0;

    for (const auto& problem : test_problems) {
        // Test if macro matches this problem
        if (!macro->match(problem)) continue;

        // Compute energy before
        double energy_before = energy.computeTotal(problem).total();

        // Apply macro
        GraphDelta delta = macro->apply(problem);
        if (delta.empty()) continue;

        Graph after_macro = problem.clone();
        after_macro.applyDelta(delta);
        double energy_after_macro = energy.computeTotal(after_macro).total();

        double macro_reduction = energy_before - energy_after_macro;

        // Compare with individual step application
        Graph after_steps = problem.clone();
        double steps_reduction = 0.0;
        int step_count = 0;
        auto applicable = baseline.getApplicable(after_steps);
        for (Transform* t : applicable) {
            if (t->match(after_steps)) {
                GraphDelta step_delta = t->apply(after_steps);
                if (!step_delta.empty()) {
                    double before = energy.computeTotal(after_steps).total();
                    after_steps.applyDelta(step_delta);
                    double after = energy.computeTotal(after_steps).total();
                    steps_reduction += (before - after);
                    step_count++;
                }
            }
        }

        // Energy equivalence: how close is macro to component steps
        if (steps_reduction > 0.001) {
            total_equivalence += macro_reduction / steps_reduction;
        } else {
            total_equivalence += 1.0;
        }

        if (macro_reduction > 0.0) {
            result.problems_improved++;
        }
    }

    result.energy_equivalence = total_equivalence / result.problems_tested;

    // Recommend if: compression > γ AND generalizes
    double generalization_rate =
        static_cast<double>(result.problems_improved) / result.problems_tested;
    result.recommended = (result.compression_ratio >= compression_threshold_) &&
                          (generalization_rate >= min_generalization_);

    return result;
}

} // namespace sare
