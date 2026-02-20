#pragma once

#include "graph/graph.hpp"
#include <string>
#include <cstdint>

namespace sare {

// ─── Graph Signature ───────────────────────────────────────────
// Extracts structural fingerprints for graph similarity lookup.
// Used by StrategyMemory to match current problems against
// previously-solved ones.
//
// The signature encodes:
// - Node type distribution (histogram)
// - Edge type distribution (histogram)
// - Degree distribution stats (mean, max)
// - Graph size (node count, edge count)

class GraphSignature {
public:
    /// Compute a structural signature string for a graph.
    /// Two graphs with similar structure will have similar signatures.
    static std::string compute(const Graph& g);

    /// Compute similarity between two signatures (0.0 = different, 1.0 = identical).
    static double similarity(const std::string& a, const std::string& b);

private:
    /// Hash a type distribution into a compact string.
    static std::string hashTypeDistribution(
        const std::unordered_map<std::string, int>& dist);
};

} // namespace sare
