#pragma once
// Shared combat inspection engine for native GPU/CPU drivers (combat_explain, combat_lab) and browser WASM.
// Hashes (observation and action keys) cross JSON as decimal strings so JavaScript preserves 64-bit precision.
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/search/config.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <unordered_map>

namespace stsrl::inspection {
using Json = nlohmann::json;
using Predict = std::function<std::vector<infra::Prediction>(std::span<const infra::Tokens>)>;

// The fight from `start` after `moves` (action bits), forced moves included; throws on an illegal move.
player::Fight replay(const Json& start, const Json& moves);
// Parses and validates search settings; `considered_actions` and `world_draw` default to Gumbel::Settings.
search::SearchConfig search_config(const Json& settings);

Json move_json(search::Action, const sts::BattleContext&);  // bits and description
Json edge_json(const search::Gumbel::Graph::Edge&);
// Fills `views` with combat::view for any node in `graph` not yet cached (such as terminal wins/losses).
void fill_terminal_views(const search::Gumbel&, const search::Gumbel::Graph&,
                         std::unordered_map<std::uint64_t, Json>& views);
// Root edges carry `bits` and `description`; when `views` is populated, nodes carry `view` and interior edges
// carry `description`.
Json graph_json(const search::Gumbel::Graph&, std::span<const search::Action> root_actions,
                const sts::BattleContext&, const std::unordered_map<std::uint64_t, Json>& views = {});
Json root_json(const search::Gumbel::Graph&, int requests);

// Stateful inspection session: keeps open searches alive between commands so scheduled Gumbel searches and
// steered simulation waves share one graph, one public-board view cache, and one batched evaluator.
// Commands (JSON objects with "op"):
//   {"op": "open", "trees": [{"start", "moves", "search"?, "draw"?, "include_graph"?}]}
//       -> {"trees": [{"id", "steps", "over", "observation", "view", ...}]}
//   {"op": "simulate", "picks": {"<id>": [[path_or_move, world, descend?], ...]}, "include_graph"?}
//       -> {"roots": {"<id>": root}, ...}
//   {"op": "graph", "tree": id}
//       -> {"decision", "network_requests", "view", "root", "graph"}
//   {"op": "state", "tree": id, "path": [edge, ...], "world": w, "worlds"?: k}
//       -> {"view", "moves", "observation", "reaching"?: [world, ...]}
//   {"op": "close", "trees": [id, ...]} -> {}
class Session {
public:
    Session(nn::Encoding, Predict);
    ~Session();
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    Json run(const Json& command);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::inspection
