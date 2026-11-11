#pragma once
// Combat as a player sees it: hidden information (draw order, future rolls) is only ever sampled, and moves
// and states are identified by what is public, so a search can share one tree across sampled worlds.
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <cstdint>
#include <span>
#include <vector>

namespace stsrl::combat {
using sts::search::Action;

// One action per key. A move is what it chooses, not where that sits: identical cards, wherever they are
// in their pile, are one move, and a key names the same choice in any state whose piles are ordered differently.
std::vector<Action> legal_actions(const sts::BattleContext&);
std::uint64_t action_key(const sts::BattleContext&, Action);
std::uint64_t observation_key(const sts::BattleContext&);

// Where a card selection (inputState CARD_SELECT) picks from. Offered choices (Codex, Discovery, Wish) come
// from cardSelectInfo rather than a pile, so they hold no cards here.
enum class Pile { hand, draw, discard, exhaust, offered };
Pile selection_pile(const sts::BattleContext&);
std::span<const sts::CardInstance> cards_in(const sts::BattleContext&, Pile);

// Worlds consistent with what `observed` shows, drawn deterministically from `stream`.
std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count, std::uint64_t stream);
std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count);  // stream: the observation
}  // namespace stsrl::combat
