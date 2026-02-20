#include "plasticity/sandbox_runner.hpp"

namespace sare {

double SandboxRunner::solveOne(const Graph& problem,
                                const EnergyAggregator& energy,
                                const TransformRegistry& registry,
                                const SearchConfig& config) {
    BeamSearch search;
    SearchResult result = search.search(problem, energy, registry, config);
    return result.best_state.energy.total();
}

SandboxResult SandboxRunner::evaluate(
    Transform* candidate,
    const std::vector<Graph>& test_problems,
    const EnergyAggregator& energy,
    const TransformRegistry& baseline_registry,
    const SearchConfig& search_config) {

    SandboxResult result;
    result.candidate_name = candidate ? candidate->name() : "";
    result.problems_tested = static_cast<int>(test_problems.size());

    if (!candidate || test_problems.empty()) return result;

    double total_baseline = 0.0;
    double total_with_candidate = 0.0;

    TransformRegistry augmented_registry;
    for (Transform* existing : baseline_registry.getAll()) {
        if (!existing) continue;
        augmented_registry.registerRaw(existing);
    }
    if (candidate && !augmented_registry.getByName(candidate->name())) {
        augmented_registry.registerRaw(candidate);
    }

    for (const auto& problem : test_problems) {
        double baseline_energy = solveOne(problem, energy, baseline_registry, search_config);
        double augmented_energy = solveOne(problem, energy, augmented_registry, search_config);

        total_baseline += baseline_energy;
        total_with_candidate += augmented_energy;

        if (augmented_energy < baseline_energy) {
            result.problems_improved++;
        }
    }

    double avg_baseline = total_baseline / test_problems.size();
    double avg_augmented = total_with_candidate / test_problems.size();

    result.performance_delta = avg_baseline - avg_augmented;
    result.avg_energy_reduction = result.performance_delta;
    result.promoted = (result.performance_delta > promotion_threshold_);

    return result;
}

} // namespace sare
