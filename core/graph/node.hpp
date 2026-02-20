#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace sare {

/// A node in the Structured Cognitive State Graph.
/// Represents a semantic unit with type, attributes, activation, and uncertainty.
struct Node {
    uint64_t id = 0;
    std::string type;
    std::unordered_map<std::string, std::string> attributes;
    double activation = 0.0;
    double uncertainty = 1.0;
    std::unordered_map<std::string, std::string> metadata;

    Node() = default;
    Node(uint64_t id, std::string type)
        : id(id), type(std::move(type)) {}

    void setAttribute(const std::string& key, const std::string& value) {
        attributes[key] = value;
    }

    std::string getAttribute(const std::string& key, const std::string& default_val = "") const {
        auto it = attributes.find(key);
        return it != attributes.end() ? it->second : default_val;
    }

    bool hasAttribute(const std::string& key) const {
        return attributes.count(key) > 0;
    }
};

} // namespace sare
