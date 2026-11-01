#pragma once
#include "infra/inference/backend.hpp"
#include <functional>
#include <nlohmann/json.hpp>

namespace stsrl::nn {
using Network = std::function<infra::Prediction(const infra::Tokens&)>;

// One decision of the agent with its whole search tree, as JSON for the Gumbel tree viewer. The request is
// {"start": combat_v4 start, "moves": [action bits], "search": {"sims", "particles", "gumbel_scale", "rollout_mix"}}:
// the moves chosen so far, with forced moves played between them as nn_play does.
nlohmann::json explain(const nlohmann::json& request, const Network&);
}  // namespace stsrl::nn
