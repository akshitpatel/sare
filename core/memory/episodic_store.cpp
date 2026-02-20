#include "memory/episodic_store.hpp"
#include <sstream>
#include <algorithm>

namespace sare {

void EpisodicStore::store(const SolveEpisode& episode) {
    episodes_.push_back(episode);
}

std::vector<SolveEpisode> EpisodicStore::retrieve(size_t limit,
                                                   bool successes_only) const {
    std::vector<SolveEpisode> result;
    for (const auto& ep : episodes_) {
        if (successes_only && !ep.success) continue;
        result.push_back(ep);
        if (limit > 0 && result.size() >= limit) break;
    }
    return result;
}

std::vector<SolveEpisode> EpisodicStore::retrieveRecent(size_t n) const {
    if (n >= episodes_.size()) return episodes_;
    return std::vector<SolveEpisode>(episodes_.end() - n, episodes_.end());
}

double EpisodicStore::successRate() const {
    if (episodes_.empty()) return 0.0;
    int successes = 0;
    for (const auto& ep : episodes_) {
        if (ep.success) successes++;
    }
    return static_cast<double>(successes) / episodes_.size();
}

double EpisodicStore::avgEnergyReduction() const {
    int count = 0;
    double total_reduction = 0.0;
    for (const auto& ep : episodes_) {
        if (ep.success) {
            total_reduction += (ep.initial_energy - ep.final_energy);
            count++;
        }
    }
    return (count > 0) ? total_reduction / count : 0.0;
}

void EpisodicStore::exportToFile(const std::string& path) const {
    std::ofstream out(path);
    for (const auto& ep : episodes_) {
        // Simple CSV-style export: problem_id, success, initial_E, final_E, time, expansions, num_transforms
        out << ep.problem_id << ","
            << (ep.success ? 1 : 0) << ","
            << ep.initial_energy << ","
            << ep.final_energy << ","
            << ep.compute_time_seconds << ","
            << ep.total_expansions << ","
            << ep.transform_sequence.size();
        // Transform sequence
        for (const auto& t : ep.transform_sequence) {
            out << "," << t;
        }
        out << "\n";
    }
}

void EpisodicStore::importFromFile(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        SolveEpisode ep;
        std::string token;

        std::getline(iss, ep.problem_id, ',');

        std::getline(iss, token, ',');
        ep.success = (token == "1");

        std::getline(iss, token, ',');
        ep.initial_energy = std::stod(token);

        std::getline(iss, token, ',');
        ep.final_energy = std::stod(token);

        std::getline(iss, token, ',');
        ep.compute_time_seconds = std::stod(token);

        std::getline(iss, token, ',');
        ep.total_expansions = std::stoi(token);

        std::getline(iss, token, ',');
        size_t num_transforms = std::stoul(token);

        for (size_t i = 0; i < num_transforms; i++) {
            if (std::getline(iss, token, ',')) {
                ep.transform_sequence.push_back(token);
            }
        }

        episodes_.push_back(ep);
    }
}

} // namespace sare
