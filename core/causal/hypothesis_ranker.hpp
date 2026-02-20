#pragma once

#include "causal/intervention.hpp"
#include <vector>
#include <algorithm>
#include <string>

namespace sare {

// ─── Causal Hypothesis ────────────────────────────────────────
// A proposed causal model with associated cost.
//
// Selection criterion:
// argmin(PredictionError + λ · Complexity)

struct CausalHypothesis {
    std::string name;
    std::vector<Intervention> interventions;
    double prediction_error = 0.0;
    double complexity = 0.0;
    double score = 0.0;  // prediction_error + λ * complexity
};

// ─── Hypothesis Ranker ────────────────────────────────────────
// Selects minimal-complexity causal model using Occam penalty.
// Implements: argmin(PredictionError + λ · Complexity)

class HypothesisRanker {
public:
    HypothesisRanker(double lambda = 1.0) : lambda_(lambda) {}

    /// Score and rank hypotheses.
    /// Returns sorted list (best first based on Occam criterion).
    std::vector<CausalHypothesis> rank(
        std::vector<CausalHypothesis>& hypotheses) const;

    /// Score a single hypothesis.
    double scoreHypothesis(const CausalHypothesis& h) const {
        return h.prediction_error + lambda_ * h.complexity;
    }

    /// Get the best hypothesis.
    CausalHypothesis best(std::vector<CausalHypothesis>& hypotheses) const;

    /// Set Occam penalty weight.
    void setLambda(double lambda) { lambda_ = lambda; }
    double lambda() const { return lambda_; }

private:
    double lambda_;  // Occam penalty weight
};

} // namespace sare
