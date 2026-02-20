#pragma once

#include "abstraction/macro_builder.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sare {

// ─── Abstraction Registry ──────────────────────────────────────
// Versioned catalog of promoted abstractions A_t.
// Abstractions are macro-transforms that have been validated
// and promoted for general use.

class AbstractionRegistry {
public:
    AbstractionRegistry() = default;

    /// Promote a validated macro-transform into the registry.
    void promote(std::unique_ptr<MacroTransform> macro,
                 const std::string& domain = "general");

    /// Demote/remove an abstraction if performance drops.
    bool demote(const std::string& name);

    /// Get all abstractions for a domain.
    std::vector<MacroTransform*> getForDomain(const std::string& domain) const;

    /// Get all registered abstractions.
    std::vector<MacroTransform*> getAll() const;

    /// Lookup by name.
    MacroTransform* getByName(const std::string& name) const;

    /// |A_t| — total abstraction count.
    size_t count() const { return abstractions_.size(); }

    /// Save registry to file (for persistence across sessions).
    void save(const std::string& path) const;

    /// Load registry from file.
    void load(const std::string& path);

    /// Clear all abstractions.
    void clear() { abstractions_.clear(); domain_map_.clear(); }

private:
    std::unordered_map<std::string, std::unique_ptr<MacroTransform>> abstractions_;
    std::unordered_map<std::string, std::vector<std::string>> domain_map_;
};

} // namespace sare
