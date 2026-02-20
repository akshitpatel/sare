#include "abstraction/trace_miner.hpp"
#include <algorithm>
#include <sstream>

namespace sare {

void TraceMiner::addTrace(const std::vector<std::string>& transform_sequence) {
    traces_.push_back(transform_sequence);
    energy_traces_.push_back({});
}

void TraceMiner::addTraceWithEnergy(
    const std::vector<std::string>& transform_sequence,
    const std::vector<double>& energy_trajectory) {
    traces_.push_back(transform_sequence);
    energy_traces_.push_back(energy_trajectory);
}

std::vector<std::vector<std::string>> TraceMiner::extractKgrams(
    const std::vector<std::string>& seq, int k) const {
    std::vector<std::vector<std::string>> result;
    if (static_cast<int>(seq.size()) < k) return result;
    for (size_t i = 0; i <= seq.size() - k; i++) {
        result.emplace_back(seq.begin() + i, seq.begin() + i + k);
    }
    return result;
}

std::string TraceMiner::subsequenceKey(
    const std::vector<std::string>& subseq) const {
    std::ostringstream oss;
    for (size_t i = 0; i < subseq.size(); i++) {
        if (i > 0) oss << "\x1F";  // unit separator
        oss << subseq[i];
    }
    return oss.str();
}

std::vector<TransformPattern> TraceMiner::mine(
    int min_frequency, int min_length, int max_length) const {

    // Count all k-gram occurrences across traces
    // key → (count, original subsequence)
    std::unordered_map<std::string, int> pattern_counts;
    std::unordered_map<std::string, std::vector<std::string>> pattern_seqs;

    for (const auto& trace : traces_) {
        for (int k = min_length; k <= max_length; k++) {
            auto kgrams = extractKgrams(trace, k);
            std::unordered_map<std::string, bool> seen;
            for (const auto& gram : kgrams) {
                std::string key = subsequenceKey(gram);
                if (!seen[key]) {
                    pattern_counts[key]++;
                    seen[key] = true;
                    if (pattern_seqs.find(key) == pattern_seqs.end()) {
                        pattern_seqs[key] = gram;
                    }
                }
            }
        }
    }

    // Build results for patterns above threshold
    std::vector<TransformPattern> results;
    for (const auto& [key, count] : pattern_counts) {
        if (count < min_frequency) continue;

        TransformPattern pattern;
        pattern.transform_subsequence = pattern_seqs[key];
        pattern.frequency = count;
        pattern.avg_compression_ratio =
            static_cast<double>(pattern.transform_subsequence.size());

        results.push_back(pattern);
    }

    // Sort by frequency (descending)
    std::sort(results.begin(), results.end(),
              [](const TransformPattern& a, const TransformPattern& b) {
                  return a.frequency > b.frequency;
              });

    return results;
}

std::vector<TransformPattern> TraceMiner::mineWithCompression(
    int min_frequency, double min_compression,
    int min_length, int max_length) const {

    auto patterns = mine(min_frequency, min_length, max_length);

    std::vector<TransformPattern> filtered;
    for (const auto& p : patterns) {
        if (p.avg_compression_ratio >= min_compression) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

} // namespace sare
