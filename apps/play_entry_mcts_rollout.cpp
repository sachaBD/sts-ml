#include "agents/mcts_agent.hpp"
#include "scenarios/slime_entry_projection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <nlohmann/json.hpp>

namespace {

void observe_split_hp(
    const stsrl::EncodedCombatState& state,
    const int boss_id,
    int& split_hp) {
    if (split_hp >= 0 || state.monsters.size() < 2
        || std::ranges::any_of(state.monsters, [boss_id](const auto& monster) {
               return monster.monster_id == boss_id;
           })) {
        return;
    }

    const auto hp = static_cast<int>(
        std::lround(state.monsters.front().numeric[0] * 100.0F));
    const bool children_agree = std::ranges::all_of(
        state.monsters, [hp](const auto& monster) {
            return static_cast<int>(
                       std::lround(monster.numeric[0] * 100.0F))
                == hp;
        });
    if (children_agree && hp > 0 && hp <= 70) {
        split_hp = hp;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: play_entry_mcts_rollout source.jsonl entry_id_or_deck_signature combat_seed simulations\n";
        return 2;
    }
    int skipped = 0;
    const auto entries = stsrl::scenarios::load_slime_entry_projections(argv[1], skipped);
    const std::string selector = argv[2];
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) { return entry.entry_id == selector || entry.deck_signature == selector; });
    if (found == entries.end()) throw std::invalid_argument{"entry selector not found"};
    const auto seed = std::stoull(argv[3]);
    const auto simulations = std::stoull(argv[4]);
    auto environment = stsrl::scenarios::slime_entry_projection(*found, seed);
    stsrl::MctsAgent agent{seed, {.simulations = simulations, .rollout_limit = 512}};

    const auto start = std::chrono::steady_clock::now();
    std::size_t decisions = 0;
    int split_hp = -1;
    const auto boss_id =
        environment.decision().encoding.monsters.front().monster_id;

    while (!environment.done()) {
        observe_split_hp(environment.decision().encoding, boss_id, split_hp);
        environment.step(agent.choose_action(environment));
        ++decisions;
    }

    std::cout
        << nlohmann::json{
               {"mode", "rollout"}, {"projection", "deck_hp_only"}, {"entry_id", found->entry_id}, {"deck_signature", found->deck_signature}, {"source_seed", found->source_seed}, {"combat_seed", seed}, {"starting_hp", found->hp}, {"starting_max_hp", found->max_hp},
               {"won", environment.won()},
               {"final_hp", environment.player_hp()},
               {"max_hp", environment.player_max_hp()},
               {"split_hp", split_hp},
               {"decisions", decisions},
               {"elapsed_seconds",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start)
                    .count()},
           }
               .dump()
        << '\n';
}
