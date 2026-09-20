#include "agents/mcts_agent.hpp"
#include "scenarios/jaw_worm.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

std::uint64_t parse_seed(const int argc, char* argv[]) {
    return argc >= 2 ? std::stoull(argv[1]) : 1;
}

std::size_t parse_simulations(const int argc, char* argv[]) {
    return argc >= 3 ? std::stoull(argv[2]) : 2'000;
}

}  // namespace

int main(const int argc, char* argv[]) try {
    const auto seed = parse_seed(argc, argv);
    const auto simulations = parse_simulations(argc, argv);
    auto environment = stsrl::scenarios::jaw_worm(seed);
    stsrl::MctsAgent agent{seed ^ 0x9e3779b97f4a7c15ULL,
                           {.simulations = simulations}};

    std::cout << "A1 Ironclad vs Jaw Worm | information-set MCTS | seed "
              << seed << " | " << simulations << " simulations/decision\n";

    std::size_t decisions = 0;
    while (!environment.done()) {
        const auto decision = environment.decision();
        const auto& state = decision.observation;
        std::cout << "\nTurn " << state.turn << " | player " << state.player_hp
                  << '/' << state.player_max_hp << " hp, " << state.player_block
                  << " block, " << state.energy << " energy | Jaw Worm "
                  << state.enemy_hp << '/' << state.enemy_max_hp << " hp, "
                  << state.enemy_block << " block\n";

        const auto chosen = agent.choose_action(environment);
        const auto refreshed = environment.decision();
        std::cout << "MCTS action: "
                  << refreshed.legal_actions.at(chosen).description << '\n';
        environment.step(chosen);
        ++decisions;
    }

    std::cout << "\nResult: " << (environment.won() ? "WIN" : "LOSS")
              << " after " << decisions << " decisions"
              << " | value " << environment.combat_value() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
