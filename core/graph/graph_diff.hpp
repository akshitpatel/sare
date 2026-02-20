#pragma once

#include "graph/graph.hpp"

namespace sare {

/// Diff utility for comparing two graph states.
/// Produces a GraphDelta that transforms graph A into graph B.
class GraphDiff {
public:
    /// Compute a delta that transforms `from` into `to`.
    static GraphDelta diff(const Graph& from, const Graph& to);

    /// Invert a delta (swap add/remove, swap before/after).
    static GraphDelta invert(const GraphDelta& delta);
};

} // namespace sare
