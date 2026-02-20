#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace sare {

// ─── Transform Pattern ────────────────────────────────────────
// A frequently-occurring subsequence of transforms detected
// across solve traces. Candidate for abstraction promotion.

struct TransformPattern {
    std::vector<std::string> transform_subsequence;
    int frequency = 0;
    double avg_compression_ratio = 0.0;  // C(P) = OrigSteps / CompressedSteps
    double avg_energy_reduction = 0.0;
};

// ─── Trace Miner ──────────────────────────────────────────────
// Detects repeated subgraph patterns across solve traces.
//
// Algorithm:
// 1. Collect solve traces T = {τ_1, τ_2, ..., τ_m}
// 2. Extract all k-length subsequences for k = 2..max_len
// 3. Count frequencies
// 4. Return patterns where freq(P) > σ
//
// These become candidates for macro transform construction.

class TraceMiner {
public:
    TraceMiner() = default;

    /// Add a solve trace (sequence of transform names).
    void addTrace(const std::vector<std::string>& transform_sequence);

    /// Add a trace with associated energy trajectory for compression evaluation.
    void addTraceWithEnergy(const std::vector<std::string>& transform_sequence,
                             const std::vector<double>& energy_trajectory);

    /// Mine frequent patterns.
    /// min_frequency: minimum occurrence count (σ)
    /// min_length: minimum subsequence length
    /// max_length: maximum subsequence length
    std::vector<TransformPattern> mine(int min_frequency = 2,
                                        int min_length = 2,
                                        int max_length = 5) const;

    /// Mine patterns that also meet compression threshold.
    /// min_compression: C(P) > γ
    std::vector<TransformPattern> mineWithCompression(
        int min_frequency = 2,
        double min_compression = 1.5,
        int min_length = 2,
        int max_length = 5) const;

    /// Number of stored traces.
    size_t traceCount() const { return traces_.size(); }

    /// Clear all stored traces.
    void clear() { traces_.clear(); energy_traces_.clear(); }

private:
    std::vector<std::vector<std::string>> traces_;
    std::vector<std::vector<double>> energy_traces_;

    /// Extract k-grams from a sequence.
    std::vector<std::vector<std::string>> extractKgrams(
        const std::vector<std::string>& seq, int k) const;

    /// Compute a hash key for a subsequence.
    std::string subsequenceKey(const std::vector<std::string>& subseq) const;
};

} // namespace sare
