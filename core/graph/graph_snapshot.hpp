#pragma once

#include "graph/graph.hpp"
#include <cstdint>
#include <vector>

namespace sare {

/// Delta-based snapshot system for search branching.
/// Instead of copying the entire graph, stores deltas from a parent state.
/// Snapshots can be restored by replaying the delta chain.

struct Snapshot {
    uint64_t id = 0;
    uint64_t parent_id = 0;  // 0 = root
    GraphDelta delta;         // delta from parent to this state
};

class SnapshotManager {
public:
    /// Take a snapshot of the current graph state.
    /// Returns the snapshot ID.
    uint64_t takeSnapshot(const GraphDelta& delta_from_parent, uint64_t parent_id = 0);

    /// Get the delta chain from root to a given snapshot.
    /// Returned in order: root → ... → snapshot.
    std::vector<GraphDelta> getDeltaChain(uint64_t snapshot_id) const;

    /// Restore a graph to a specific snapshot state.
    /// Starts from a root graph and applies the full delta chain.
    Graph restore(const Graph& root_graph, uint64_t snapshot_id) const;

    /// Get a specific snapshot.
    const Snapshot* getSnapshot(uint64_t id) const;

    size_t snapshotCount() const { return snapshots_.size(); }

private:
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, Snapshot> snapshots_;
};

} // namespace sare
