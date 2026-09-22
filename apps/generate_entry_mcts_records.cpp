#include "agents/mcts_agent.hpp"
#include "scenarios/slime_entry_projection.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

static std::uint64_t combat_seed(std::uint64_t source, std::uint64_t replicate) {
    return source ^ (0x9e3779b97f4a7c15ULL * (replicate + 1));
}
int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "usage: generate_entry_mcts_records input.jsonl simulations rollout replicates root_limit\n"; return 2; }
    int skipped = 0;
    auto entries = stsrl::scenarios::load_slime_entry_projections(argv[1], skipped);
    const auto simulations = std::stoull(argv[2]), rollout = std::stoull(argv[3]), replicates = std::stoull(argv[4]);
    entries.resize(std::min(entries.size(), static_cast<std::size_t>(std::stoull(argv[5]))));
    std::uint64_t episode = 0;
    for (const auto& entry : entries) for (std::uint64_t replicate = 0; replicate < replicates; ++replicate, ++episode) {
        const auto seed = combat_seed(entry.source_seed, replicate);
        auto env = stsrl::scenarios::slime_entry_projection(entry, seed);
        stsrl::MctsAgent agent{seed, {.simulations=simulations, .rollout_limit=rollout}};
        std::vector<nlohmann::json> rows; int decision=0;
        while (!env.done()) { auto d=env.decision(); auto result=agent.search(env); nlohmann::json row=d.encoding;
            row["entry_id"]=entry.entry_id; row["source_seed"]=entry.source_seed; row["public_snapshot_sha256"]=entry.public_snapshot_sha256; row["deck_signature"]=entry.deck_signature;
            row["starting_hp"]=entry.hp; row["starting_max_hp"]=entry.max_hp; row["seed"]=seed; row["combat_seed"]=seed; row["replicate"]=replicate; row["episode_id"]=episode; row["decision_index"]=decision++;
            row["mcts_value"]=result.root_value; row["root_visits"]=result.root_visits; row["chosen_action"]=result.chosen_action; rows.push_back(std::move(row)); env.step(result.chosen_action); }
        for (auto& row: rows) { row["terminal_outcome"]=env.won()?1:-1; row["final_player_hp"]=env.player_hp(); row["final_player_max_hp"]=env.player_max_hp(); row["terminal_value"]=env.combat_value(); auto b=nlohmann::json::to_msgpack(row); std::cout.write((char*)b.data(), b.size()); }
        std::cerr << "episode " << episode << '\n';
    }
}
