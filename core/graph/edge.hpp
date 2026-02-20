#pragma once

#include <cstdint>
#include <string>

namespace sare {

/// A directed edge in the Structured Cognitive State Graph.
/// Connects source → target with a typed relationship and weight.
struct Edge {
    uint64_t id = 0;
    uint64_t source = 0;
    uint64_t target = 0;
    std::string relationship_type;
    double weight = 1.0;

    Edge() = default;
    Edge(uint64_t id, uint64_t source, uint64_t target, std::string rel_type, double weight = 1.0)
        : id(id), source(source), target(target),
          relationship_type(std::move(rel_type)), weight(weight) {}
};

} // namespace sare
