#pragma once
// For tests: repeatable predictions that depend on the encoded query alone.
#include "agents/combat/nn/tokens.hpp"
#include <cmath>
#include <numeric>

namespace stsrl::player {
inline infra::Prediction fake_network(const infra::Tokens& t) {
    const auto row = [&](int i) { return std::accumulate(t.data.begin() + i * nn::width, t.data.begin() + (i + 1) * nn::width, 0.0); };
    infra::Prediction p{float(0.5 + 0.4 * std::sin(row(0))), {}};
    for (int i = t.count - t.actions; i < t.count; ++i) p.logits.push_back(float(std::sin(row(i))));
    return p;
}
}  // namespace stsrl::player
