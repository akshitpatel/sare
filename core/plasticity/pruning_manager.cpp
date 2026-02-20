#include "plasticity/pruning_manager.hpp"

namespace sare {

std::vector<std::string> PruningManager::identifyCandidates(
    const TransformRegistry& registry) const {

    std::vector<std::string> candidates;
    auto all_transforms = registry.getAll();

    for (Transform* t : all_transforms) {
        if (t->applicationCount() > 0 && t->getUtility() < utility_threshold_) {
            candidates.push_back(t->name());
        }
    }

    return candidates;
}

int PruningManager::prune(TransformRegistry& registry,
                           const std::vector<std::string>& names) {
    int removed = 0;
    for (const auto& name : names) {
        if (registry.remove(name)) {
            removed++;
        }
    }
    return removed;
}

int PruningManager::pruneUnderperformers(TransformRegistry& registry) {
    auto candidates = identifyCandidates(registry);
    return prune(registry, candidates);
}

} // namespace sare
