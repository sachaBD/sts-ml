#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>

namespace stsrl::pv {

// AlphaZero-style Q + U. Q is the backed-up mean, or zero for an unvisited edge.
// The +1 parent pseudovisit makes priors steer the very first traversal.
// In-flight simulations count as temporary zero-value visits (virtual loss).
struct PuctScore {
    const double exploration = 150;  // HP-equivalent points; a starting setting, not a tuned result

    PuctScore() = default;
    explicit PuctScore(double points) : exploration{points} {
        if (!std::isfinite(points) || points < 0) throw std::invalid_argument{"PV: invalid PUCT coefficient"};
    }

    double operator()(double q, double prior, std::int64_t parent_visits, std::int64_t edge_visits) const {
        return q + exploration * prior * std::sqrt(1. + parent_visits) / (1 + edge_visits);
    }
};

struct ArgmaxSelection {
    std::size_t operator()(std::span<const double> scores, std::mt19937_64&) const {
        if (scores.empty()) throw std::logic_error{"PV: cannot select from an empty node"};
        for (double score : scores) {
            if (!std::isfinite(score)) throw std::runtime_error{"PV: non-finite selection score"};
        }

        // Stable ties follow legal-action order; there is no implicit shuffle or policy sampling.
        return std::max_element(scores.begin(), scores.end()) - scores.begin();
    }
};

}  // namespace stsrl::pv
