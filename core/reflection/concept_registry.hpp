#pragma once

#include "reflection_engine.hpp"
#include <vector>
#include <string>
#include <algorithm>

namespace sare {

class ConceptRegistry {
public:
    ConceptRegistry() = default;

    /// Register a discovered rule.
    /// If a similar rule exists, merge/reinforce it.
    void addRule(const AbstractRule& rule) {
        // Simple check for duplicates based on name/pattern size
        // Real implementation would use graph isomorphism check
        for (auto& existing : rules_) {
            if (existing.name == rule.name &&
                existing.pattern.nodeCount() == rule.pattern.nodeCount()) {
                existing.observations++;
                existing.confidence = std::min(1.0, existing.confidence + 0.1);
                return;
            }
        }
        rules_.push_back(rule);
    }

    /// Retrieve all learned rules.
    const std::vector<AbstractRule>& getRules() const {
        return rules_;
    }

    /// Retrieve rules with high confidence.
    std::vector<AbstractRule> getConsolidatedRules(double min_confidence = 0.8) const {
        std::vector<AbstractRule> consolidated;
        for (const auto& rule : rules_) {
            if (rule.confidence >= min_confidence) {
                consolidated.push_back(rule);
            }
        }
        return consolidated;
    }

private:
    std::vector<AbstractRule> rules_;
};

} // namespace sare
