#include "causal/hypothesis_ranker.hpp"

namespace sare {

std::vector<CausalHypothesis> HypothesisRanker::rank(
    std::vector<CausalHypothesis>& hypotheses) const {

    // Score each hypothesis
    for (auto& h : hypotheses) {
        h.score = scoreHypothesis(h);
    }

    // Sort by score (lower = better, Occam's razor)
    std::sort(hypotheses.begin(), hypotheses.end(),
              [](const CausalHypothesis& a, const CausalHypothesis& b) {
                  return a.score < b.score;
              });

    return hypotheses;
}

CausalHypothesis HypothesisRanker::best(
    std::vector<CausalHypothesis>& hypotheses) const {

    if (hypotheses.empty()) return CausalHypothesis{};
    auto ranked = rank(hypotheses);
    return ranked[0];
}

} // namespace sare
