#include "agents/mcts_agent.hpp"
#include "scenarios/slime_entry_projection.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: smoke_slime_entry_projections input.jsonl output.jsonl simulations rollout_limit\n";
        return 2;
    }
    int skipped = 0;
    const auto entries = stsrl::scenarios::load_slime_entry_projections(argv[1], skipped);
    std::ofstream output{argv[2]};
    if (!output) throw std::runtime_error{"cannot open smoke output"};
    for (std::size_t ordinal = 0; ordinal < entries.size(); ++ordinal) {
        const auto& entry = entries[ordinal];
        nlohmann::json row{{"schema_version", 1}, {"kind", "slime_entry_projection_smoke"},
            {"projection", "deck_hp_only_relics_potions_bottles_gold_map_cleared"},
            {"entry_id", entry.entry_id}, {"source_seed", entry.source_seed},
            {"starting_hp", entry.hp}, {"starting_max_hp", entry.max_hp},
            {"public_snapshot_sha256", entry.public_snapshot_sha256}, {"deck_signature", entry.deck_signature},
            {"deck_card_ids", entry.card_ids}, {"deck_upgrades", entry.upgrades}, {"deck_misc", entry.misc},
            {"skipped_nonaccepted_input_rows", skipped}};
        try {
            auto environment = stsrl::scenarios::slime_entry_projection(entry);
            stsrl::MctsAgent agent{entry.source_seed, {.simulations = std::stoull(argv[3]), .rollout_limit = std::stoull(argv[4])}};
            int decisions = 0;
            while (!environment.done()) {
                environment.step(agent.choose_action(environment));
                ++decisions;
            }
            row["status"] = "completed";
            row["outcome"] = environment.won() ? "win" : "loss";
            row["decisions"] = decisions;
            row["final_hp"] = environment.player_hp();
            row["final_max_hp"] = environment.player_max_hp();
        } catch (const std::exception& error) {
            row["status"] = "failure";
            row["error"] = error.what();
        }
        output << row.dump() << '\n';
        output.flush();
        std::cerr << '[' << ordinal + 1 << '/' << entries.size() << "] " << entry.entry_id << ' ' << row["status"] << '\n';
    }
}
