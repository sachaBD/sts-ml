#pragma once
// A finished search as compact JSON, from which the search at any simulation count can be rebuilt.
//   nodes:   [value, terminal, [logits...]] per node; edges in node order; priors are softmax(logits)
//   root:    index into nodes; root_actions/root_descriptions/root_noise in root edge order
//   backups: per backup in search order, [edge, reached, edge, reached, ...] from the root; the last
//            reached node is the backed-up value's node (the root, for an empty path)
#include "agents/combat/search/gumbel/gumbel.hpp"
#include <nlohmann/json.hpp>

namespace stsrl::player {
nlohmann::json trace(const search::Gumbel::Graph&, const search::Result&, const sts::BattleContext&);
}  // namespace stsrl::player
