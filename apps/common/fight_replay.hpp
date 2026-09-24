// Rebuilding one stored fight of a combat_v3 bootstrap run (the request: apps/common/replay.py), shared by
// apps/value_play, apps/dagger and apps/fight_resample:
// the act 1 run is replayed up to that fight (SimpleAgent out of combat, as in bootstrap; the earlier
// fights by their stored chosen actions), then `play` plays the fight and the run stops.
#pragma once

#include "combat/environment.hpp"
#include "constants/CharacterClasses.h"
#include "scenarios/act1_run.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl::replay {

// Plays a stored fight; it must end exactly after its last action.
inline sts::BattleContext stored_fight(const sts::BattleContext& start, const std::vector<std::size_t>& actions) {
    CombatEnvironment env{start};
    for (const auto action : actions) {
        if (env.done() || action >= env.decision().legal_actions.size())
            throw std::runtime_error{"stored actions don't fit the replayed fight"};
        env.step(action);
    }
    if (!env.done()) throw std::runtime_error{"replayed fight didn't end after its stored actions"};
    return env.battle();
}

// As act1::FightFn, plus the game right before the fight (`start` was built from it by BattleContext::init).
using GameFightFn = std::function<sts::BattleContext(const sts::GameContext& game, const sts::BattleContext& start,
                                                     const nlohmann::json& fight)>;

// input: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight]}.
// Returns the target fight's combat_v3 fight columns (episode_id, ...).
inline nlohmann::json to_fight(const nlohmann::json& input, const GameFightFn& play) {
    const auto actions = input.at("actions").get<std::vector<std::vector<std::size_t>>>();
    const auto target = input.at("fight_index").get<int>();
    if (static_cast<int>(actions.size()) != target) throw std::invalid_argument{"need actions for every earlier fight"};
    sts::GameContext game{sts::CharacterClass::IRONCLAD, input.at("run_seed").get<std::uint64_t>(),
                          input.at("ascension").get<int>()};
    nlohmann::json fight_columns;
    const auto run = act1::play(game, [&](const sts::BattleContext& start, const nlohmann::json& fight) {
        const auto index = fight.at("fight_index").get<int>();
        if (index < target) return stored_fight(start, actions[index]);
        fight_columns = fight;
        return play(game, start, fight);
    }, target + 1);
    if (run.fights != target + 1) throw std::runtime_error{"run ended before the fight"};
    return fight_columns;
}

inline nlohmann::json to_fight(const nlohmann::json& input, const act1::FightFn& play) {
    return to_fight(input, GameFightFn{[&](const sts::GameContext&, const sts::BattleContext& start,
                                           const nlohmann::json& fight) { return play(start, fight); }});
}

}  // namespace stsrl::replay
