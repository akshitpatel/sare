#include "abstraction/abstraction_registry.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace sare {

void AbstractionRegistry::promote(std::unique_ptr<MacroTransform> macro,
                                   const std::string& domain) {
    std::string name = macro->name();
    domain_map_[domain].push_back(name);
    abstractions_[name] = std::move(macro);
}

bool AbstractionRegistry::demote(const std::string& name) {
    auto it = abstractions_.find(name);
    if (it == abstractions_.end()) return false;

    // Remove from domain maps
    for (auto& [domain, names] : domain_map_) {
        names.erase(std::remove(names.begin(), names.end(), name), names.end());
    }

    abstractions_.erase(it);
    return true;
}

std::vector<MacroTransform*> AbstractionRegistry::getForDomain(
    const std::string& domain) const {
    std::vector<MacroTransform*> result;
    auto it = domain_map_.find(domain);
    if (it != domain_map_.end()) {
        for (const auto& name : it->second) {
            auto abs_it = abstractions_.find(name);
            if (abs_it != abstractions_.end()) {
                result.push_back(abs_it->second.get());
            }
        }
    }
    return result;
}

std::vector<MacroTransform*> AbstractionRegistry::getAll() const {
    std::vector<MacroTransform*> result;
    for (const auto& [name, macro] : abstractions_) {
        result.push_back(macro.get());
    }
    return result;
}

MacroTransform* AbstractionRegistry::getByName(const std::string& name) const {
    auto it = abstractions_.find(name);
    return (it != abstractions_.end()) ? it->second.get() : nullptr;
}

void AbstractionRegistry::save(const std::string& path) const {
    std::ofstream out(path);
    for (const auto& [name, macro] : abstractions_) {
        // Format: name|step1,step2,step3|domain
        out << name << "|";
        const auto& steps = macro->stepNames();
        for (size_t i = 0; i < steps.size(); i++) {
            if (i > 0) out << ",";
            out << steps[i];
        }
        // Find domain
        std::string domain = "general";
        for (const auto& [d, names] : domain_map_) {
            for (const auto& n : names) {
                if (n == name) { domain = d; break; }
            }
        }
        out << "|" << domain << "\n";
    }
}

void AbstractionRegistry::load(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        // Parse: name|step1,step2|domain
        std::istringstream iss(line);
        std::string name, steps_str, domain;
        std::getline(iss, name, '|');
        std::getline(iss, steps_str, '|');
        std::getline(iss, domain, '|');

        // Parse steps
        std::vector<std::string> step_names;
        std::istringstream steps_iss(steps_str);
        std::string step;
        while (std::getline(steps_iss, step, ',')) {
            step_names.push_back(step);
        }

        auto macro = std::make_unique<MacroTransform>(name, step_names);
        promote(std::move(macro), domain);
    }
}

} // namespace sare
