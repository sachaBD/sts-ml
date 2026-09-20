#include "agents/mcts_agent.hpp"
#include "combat/encoding.hpp"
#include "scenarios/slime_boss.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: generate_mcts_records seed count simulations rollout exploration\n";
        return 2;
    }

    const auto first = std::stoull(argv[1]);
    const auto count = std::stoull(argv[2]);
    const stsrl::MctsConfig config{
        .simulations = std::stoull(argv[3]),
        .rollout_limit = std::stoull(argv[4]),
        .exploration = std::stod(argv[5]),
    };

    for (std::uint64_t episode = 0; episode < count; ++episode) {
        const auto seed = first + episode;
        auto env = stsrl::scenarios::slime_boss(seed);
        stsrl::MctsAgent agent{seed, config};
        std::vector<nlohmann::json> rows;
        int decision_index = 0;

        while (!env.done()) {
            const auto decision = env.decision();
            const auto result = agent.search(env);

            nlohmann::json row = decision.encoding;
            row["episode_id"] = episode;
            row["seed"] = seed;
            row["decision_index"] = decision_index++;
            row["mcts_value"] = result.root_value;
            row["root_visits"] = result.root_visits;
            row["chosen_action"] = result.chosen_action;
            rows.push_back(std::move(row));

            env.step(result.chosen_action);
        }

        const int outcome = env.won() ? 1 : -1;
        const auto value = env.combat_value();
        for (auto& row : rows) {
            row["terminal_outcome"] = outcome;
            row["final_player_hp"] = env.player_hp();
            row["final_player_max_hp"] = env.player_max_hp();
            row["terminal_value"] = value;

            const auto bytes = nlohmann::json::to_msgpack(row);
            std::cout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        std::cerr << "episode " << episode + 1 << '/' << count << '\n';
    }

    return 0;
}
