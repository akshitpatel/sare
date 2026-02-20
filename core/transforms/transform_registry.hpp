#pragma once

#include "transforms/transform_base.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sare {

/// Registry for all transformation modules.
/// Manages registration, activation thresholds, and utility-based filtering.
class TransformRegistry {
public:
    /// Register a new transform module (takes ownership).
    void registerTransform(std::unique_ptr<Transform> transform);

    /// Register a raw pointer (non-owning). Used for sandbox evaluation.
    void registerRaw(Transform* transform);

    /// Remove a transform by name. Returns true if found.
    bool remove(const std::string& name);

    /// Get all transforms applicable to the current graph state,
    /// filtered by match() and utility threshold.
    std::vector<Transform*> getApplicable(const Graph& graph, double utility_threshold = 0.0) const;

    /// Get all registered transforms (unfiltered).
    std::vector<Transform*> getAll() const;

    /// Look up a transform by name.
    Transform* getByName(const std::string& name) const;

    /// Update a transform's utility after application.
    void recordApplication(const std::string& name, double actual_delta_energy);

    size_t count() const { return transforms_.size() + raw_transforms_.size(); }

private:
    std::vector<std::unique_ptr<Transform>> transforms_;
    std::vector<Transform*> raw_transforms_;  // non-owning
    std::unordered_map<std::string, size_t> name_index_;
};

} // namespace sare
