#include "graph/graph_snapshot.hpp"
#include <algorithm>
#include <stdexcept>

namespace sare {

uint64_t SnapshotManager::takeSnapshot(const GraphDelta& delta_from_parent, uint64_t parent_id) {
    Snapshot snap;
    snap.id = next_id_++;
    snap.parent_id = parent_id;
    snap.delta = delta_from_parent;
    snapshots_[snap.id] = std::move(snap);
    return snap.id - 1 + 1; // just return snap.id clearly
}

std::vector<GraphDelta> SnapshotManager::getDeltaChain(uint64_t snapshot_id) const {
    std::vector<GraphDelta> chain;
    uint64_t current = snapshot_id;

    while (current != 0) {
        auto it = snapshots_.find(current);
        if (it == snapshots_.end()) {
            throw std::runtime_error("Snapshot not found: " + std::to_string(current));
        }
        chain.push_back(it->second.delta);
        current = it->second.parent_id;
    }

    // Reverse to get root → snapshot order
    std::reverse(chain.begin(), chain.end());
    return chain;
}

Graph SnapshotManager::restore(const Graph& root_graph, uint64_t snapshot_id) const {
    Graph g = root_graph.clone();
    auto chain = getDeltaChain(snapshot_id);
    for (const auto& delta : chain) {
        g.applyDelta(delta);
    }
    return g;
}

const Snapshot* SnapshotManager::getSnapshot(uint64_t id) const {
    auto it = snapshots_.find(id);
    return it != snapshots_.end() ? &it->second : nullptr;
}

} // namespace sare
