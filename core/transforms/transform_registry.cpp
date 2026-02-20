#include "transforms/transform_registry.hpp"
#include <algorithm>

namespace sare {

void TransformRegistry::registerTransform(std::unique_ptr<Transform> transform) {
    std::string n = transform->name();
    name_index_[n] = transforms_.size();
    transforms_.push_back(std::move(transform));
}

void TransformRegistry::registerRaw(Transform* transform) {
    if (transform) {
        raw_transforms_.push_back(transform);
    }
}

bool TransformRegistry::remove(const std::string& name) {
    auto it = name_index_.find(name);
    if (it == name_index_.end()) {
        // Check raw transforms
        for (auto rit = raw_transforms_.begin(); rit != raw_transforms_.end(); ++rit) {
            if ((*rit)->name() == name) {
                raw_transforms_.erase(rit);
                return true;
            }
        }
        return false;
    }

    size_t idx = it->second;
    transforms_.erase(transforms_.begin() + idx);
    name_index_.erase(it);

    // Rebuild index for remaining transforms
    name_index_.clear();
    for (size_t i = 0; i < transforms_.size(); i++) {
        name_index_[transforms_[i]->name()] = i;
    }
    return true;
}

std::vector<Transform*> TransformRegistry::getApplicable(
    const Graph& graph, double utility_threshold
) const {
    std::vector<Transform*> result;
    for (const auto& t : transforms_) {
        if (t->applicationCount() > 0 && t->getUtility() < utility_threshold) {
            continue;
        }
        if (t->match(graph)) {
            result.push_back(t.get());
        }
    }
    for (Transform* t : raw_transforms_) {
        if (t->applicationCount() > 0 && t->getUtility() < utility_threshold) {
            continue;
        }
        if (t->match(graph)) {
            result.push_back(t);
        }
    }
    return result;
}

std::vector<Transform*> TransformRegistry::getAll() const {
    std::vector<Transform*> result;
    for (const auto& t : transforms_) {
        result.push_back(t.get());
    }
    for (Transform* t : raw_transforms_) {
        result.push_back(t);
    }
    return result;
}

Transform* TransformRegistry::getByName(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        return transforms_[it->second].get();
    }
    for (Transform* t : raw_transforms_) {
        if (t->name() == name) return t;
    }
    return nullptr;
}

void TransformRegistry::recordApplication(const std::string& name, double actual_delta_energy) {
    Transform* t = getByName(name);
    if (t) {
        t->recordApplication(actual_delta_energy);
    }
}

} // namespace sare
