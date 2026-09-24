#include "agents/mcts_agent.hpp"
#include "scenarios/slime_boss.hpp"

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
    const bool oracle = argc == 4 && std::string{argv[3]} == "oracle";
    if (argc != 3 && !oracle) {
        std::cerr << "usage: play_slime_mcts_rollout seed simulations [oracle]\n";
        return 2;
    }

    const auto seed = std::stoull(argv[1]);
    const auto simulations = std::stoull(argv[2]);
    auto environment = stsrl::scenarios::slime_boss(seed);
    stsrl::MctsAgent agent{seed, {.simulations = simulations, .oracle = oracle}};

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
               {"seed", seed},
               {"oracle", oracle},
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
