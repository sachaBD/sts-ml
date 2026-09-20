#include "agents/random_agent.hpp"
#include "scenarios/jaw_worm.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

constexpr std::size_t max_decisions = 1'000;

std::uint64_t parse_seed(const int argc, char* argv[]) {
    if (argc < 2) {
        return 1;
    }
    return std::stoull(argv[1]);
}

}  // namespace

int main(const int argc, char* argv[]) try {
    const auto seed = parse_seed(argc, argv);
    auto environment = stsrl::scenarios::jaw_worm(seed);
    stsrl::RandomAgent agent{seed ^ 0x9e3779b97f4a7c15ULL};

    std::cout << "A1 Ironclad vs Jaw Worm (seed " << seed << ")\n";

    std::size_t decisions = 0;
    while (!environment.done() && decisions < max_decisions) {
        const auto decision = environment.decision();
        const auto& state = decision.observation;
        std::cout << "\nTurn " << state.turn << " | player " << state.player_hp
                  << '/' << state.player_max_hp << " hp, " << state.player_block
                  << " block, " << state.energy << " energy | Jaw Worm "
                  << state.enemy_hp << '/' << state.enemy_max_hp << " hp, "
                  << state.enemy_block << " block\n";

        const auto chosen = agent.choose_action(state, decision.legal_actions);
        std::cout << "Random action: "
                  << decision.legal_actions.at(chosen).description << '\n';

        environment.step(chosen);
        ++decisions;
    }

    if (!environment.done()) {
        std::cerr << "Episode exceeded " << max_decisions << " decisions\n";
        return 1;
    }

    std::cout << "\nResult: " << (environment.won() ? "WIN" : "LOSS")
              << " after " << decisions << " decisions\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
