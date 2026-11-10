#pragma once
// For tests: repeatable predictions that depend on the encoded query alone.
#include "infra/inference/backend.hpp"
#include <cmath>
#include <numeric>

namespace stsrl::player {
inline infra::Prediction fake_network(const infra::Tokens& t) {
    const auto width = int(t.data.size()) / t.count;
    const auto row = [&](int i) { return std::accumulate(t.data.begin() + i * width, t.data.begin() + (i + 1) * width, 0.0); };
    infra::Prediction p{float(0.5 + 0.4 * std::sin(row(0))), {}};
    for (int i = t.count - t.actions; i < t.count; ++i) p.logits.push_back(float(std::sin(row(i))));
    return p;
}
}  // namespace stsrl::player
