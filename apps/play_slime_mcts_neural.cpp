#include "agents/mcts_agent.hpp"
#include "scenarios/slime_boss.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

void write_frame(const nlohmann::json& message) {
    const auto bytes = nlohmann::json::to_msgpack(message);
    const auto length = htonl(static_cast<std::uint32_t>(bytes.size()));
    std::cout.write(reinterpret_cast<const char*>(&length), sizeof(length));
    std::cout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error{"failed to write neural gameplay frame"};
    }
}

double evaluate_leaf(stsrl::CombatEnvironment& environment) {
    write_frame({{"type", "leaf"}, {"state", environment.decision().encoding}});

    std::uint32_t network_bits{};
    if (!std::cin.read(reinterpret_cast<char*>(&network_bits), sizeof(network_bits))) {
        throw std::runtime_error{"missing neural leaf response"};
    }
    const float value = std::bit_cast<float>(ntohl(network_bits));
    if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
        throw std::runtime_error{"invalid neural leaf response"};
    }
    return value;
}

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
    if (argc != 3) {
        std::cerr << "usage: play_slime_mcts_neural seed simulations\n";
        return 2;
    }

    const auto seed = std::stoull(argv[1]);
    const auto simulations = std::stoull(argv[2]);
    auto environment = stsrl::scenarios::slime_boss(seed);
    stsrl::MctsAgent agent{
        seed,
        {.simulations = simulations},
        evaluate_leaf,
    };

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

    write_frame({
        {"type", "result"},
        {"seed", seed},
        {"won", environment.won()},
        {"final_hp", environment.player_hp()},
        {"max_hp", environment.player_max_hp()},
        {"split_hp", split_hp},
        {"decisions", decisions},
        {"elapsed_seconds",
         std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start)
             .count()},
    });
}
