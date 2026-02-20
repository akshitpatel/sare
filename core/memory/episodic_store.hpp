#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <optional>
#include <functional>

namespace sare {

// ─── Solve Episode ─────────────────────────────────────────────
// A complete record of one solve attempt. The fundamental unit of
// episodic memory. Stores the full transform trace and energy
// trajectory for learning and abstraction mining.

struct SolveEpisode {
    std::string problem_id;
    std::vector<std::string> transform_sequence;
    std::vector<double> energy_trajectory;  // E at each step
    double initial_energy = 0.0;
    double final_energy = 0.0;
    double compute_time_seconds = 0.0;
    int total_expansions = 0;
    bool success = false;  // true if E → ~0
};

// ─── Episodic Store ────────────────────────────────────────────
// Stores complete solve traces. The raw material for:
// - Heuristic training (Phase 2)
// - Abstraction mining (Phase 3)
// - Credit assignment
// - Performance tracking

class EpisodicStore {
public:
    EpisodicStore() = default;

    /// Store a new solve episode.
    void store(const SolveEpisode& episode);

    /// Retrieve episodes. Optional filter: successes only.
    std::vector<SolveEpisode> retrieve(size_t limit = 0,
                                        bool successes_only = false) const;

    /// Retrieve the N most recent episodes.
    std::vector<SolveEpisode> retrieveRecent(size_t n) const;

    /// Total stored episodes.
    size_t count() const { return episodes_.size(); }

    /// Success rate across all stored episodes.
    double successRate() const;

    /// Average energy reduction across successful solves.
    double avgEnergyReduction() const;

    /// Export all episodes to JSONL file.
    void exportToFile(const std::string& path) const;

    /// Import episodes from JSONL file (appends).
    void importFromFile(const std::string& path);

    /// Clear all episodes.
    void clear() { episodes_.clear(); }

private:
    std::vector<SolveEpisode> episodes_;
};

} // namespace sare
