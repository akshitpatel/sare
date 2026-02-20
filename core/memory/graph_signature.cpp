#include "memory/graph_signature.hpp"
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <functional>

namespace sare {

std::string GraphSignature::compute(const Graph& g) {
    // Collect node type distribution
    std::unordered_map<std::string, int> node_types;
    std::unordered_map<std::string, int> edge_types;
    int total_nodes = 0;
    int total_edges = 0;
    double total_degree = 0.0;
    int max_degree = 0;

    g.forEachNode([&](const Node& n) {
        node_types[n.type]++;
        total_nodes++;
        // Compute degree
        int degree = static_cast<int>(g.getOutgoing(n.id).size() +
                                       g.getIncoming(n.id).size());
        total_degree += degree;
        max_degree = std::max(max_degree, degree);
    });

    g.forEachEdge([&](const Edge& e) {
        edge_types[e.relationship_type]++;
        total_edges++;
    });

    // Build signature string: size|node_types|edge_types|degree_stats
    std::ostringstream sig;
    sig << "N" << total_nodes << "E" << total_edges << "|";

    // Sorted node type histogram
    sig << hashTypeDistribution(node_types) << "|";

    // Sorted edge type histogram
    sig << hashTypeDistribution(edge_types) << "|";

    // Degree stats
    double avg_degree = (total_nodes > 0) ? total_degree / total_nodes : 0.0;
    sig << "d" << static_cast<int>(avg_degree * 10) << "m" << max_degree;

    return sig.str();
}

double GraphSignature::similarity(const std::string& a, const std::string& b) {
    if (a == b) return 1.0;
    if (a.empty() || b.empty()) return 0.0;

    // Use a simple character-level Jaccard similarity on the signature
    // This is a fast approximation; a more sophisticated approach would
    // parse the components and compare distributions directly.

    // Split into components
    auto split = [](const std::string& s, char delim) {
        std::vector<std::string> parts;
        std::istringstream iss(s);
        std::string part;
        while (std::getline(iss, part, delim)) {
            parts.push_back(part);
        }
        return parts;
    };

    auto parts_a = split(a, '|');
    auto parts_b = split(b, '|');

    if (parts_a.size() != parts_b.size()) return 0.0;

    double total_sim = 0.0;
    int components = 0;

    for (size_t i = 0; i < parts_a.size(); i++) {
        if (parts_a[i] == parts_b[i]) {
            total_sim += 1.0;
        } else {
            // Partial match: count common characters
            std::unordered_map<char, int> chars_a, chars_b;
            for (char c : parts_a[i]) chars_a[c]++;
            for (char c : parts_b[i]) chars_b[c]++;

            int intersection = 0, union_size = 0;
            for (auto& [c, count] : chars_a) {
                intersection += std::min(count, chars_b[c]);
                union_size += count;
            }
            for (auto& [c, count] : chars_b) {
                if (chars_a.find(c) == chars_a.end()) {
                    union_size += count;
                }
            }
            total_sim += (union_size > 0)
                ? static_cast<double>(intersection) / union_size : 0.0;
        }
        components++;
    }

    return (components > 0) ? total_sim / components : 0.0;
}

std::string GraphSignature::hashTypeDistribution(
    const std::unordered_map<std::string, int>& dist) {
    // Sort by type name for deterministic output
    std::vector<std::pair<std::string, int>> sorted(dist.begin(), dist.end());
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream oss;
    for (size_t i = 0; i < sorted.size(); i++) {
        if (i > 0) oss << ",";
        oss << sorted[i].first << ":" << sorted[i].second;
    }
    return oss.str();
}

} // namespace sare
