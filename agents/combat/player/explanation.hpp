#pragma once
#include "agents/combat/nn/evaluator.hpp"
#include <functional>
#include <nlohmann/json.hpp>

namespace stsrl::player {
using Network = std::function<infra::Prediction(const infra::Tokens&)>;
// Replays start + moves (forced moves automatic), then reruns one Gumbel decision.
// Request: {start, moves: [action bits], search: {sims, particles, gumbel_scale, rollout_mix}}.
nlohmann::json explain(const nlohmann::json&, nn::Encoding, const Network&);
}  // namespace stsrl::player
