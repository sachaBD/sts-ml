#pragma once
// Shared by the inspector apps: replaying a request's fight and the search's graph as JSON.
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include <nlohmann/json.hpp>

namespace stsrl::inspection {
using Json = nlohmann::json;

// The fight from `start` after `moves` (action bits), forced moves included; throws on an illegal move.
player::Fight replay(const Json& start, const Json& moves);
Json move_json(search::Action, const sts::BattleContext&);  // bits and description
Json edge_json(const search::Gumbel::Graph::Edge&);
// Root edges also carry their move (move_json).
Json graph_json(const search::Gumbel::Graph&, std::span<const search::Action> root_actions, const sts::BattleContext&);
}  // namespace stsrl::inspection
