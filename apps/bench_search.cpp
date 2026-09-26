// Search performance benchmark (data-gen teacher: value-net leaves, 20k simulations, 8 particles, no random move).
//   bench_search play WEIGHTS FIRST COUNT [SIMS]   -> one JSON line per fight (time, sims, outcome)
// Slime Boss scenario, one core.
#include "agents/teacher_search.hpp"
#include "models/value_net.hpp"
#include "scenarios/slime_boss.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <iostream>
#include <string>

using namespace stsrl;
using Json = nlohmann::json;

namespace {
double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: bench_search play WEIGHTS FIRST COUNT [SIMS]\n";
        return 2;
    }
    const std::string mode = argv[1];
    const ValueNet net{argv[2]};
    const auto first = std::stoull(argv[3]), count = std::stoull(argv[4]);
    const std::int64_t sims = argc > 5 ? std::stoll(argv[5]) : 20000;
    const auto run = teacher::value_net_search(net, sims);
    std::unique_ptr<std::ofstream> trace;
    if (const char* path = std::getenv("TRACE")) trace = std::make_unique<std::ofstream>(path);
    if (mode == "phases") {
        // Split the value-net search loop into tree (requestBatch), encode, net, backup (submit).
        double tree = 0, encode = 0, eval = 0, backup = 0;
        std::int64_t total = 0;
        for (auto seed = first; seed < first + count; ++seed) {
            auto env = scenarios::slime_boss(seed);
            while (!env.done()) {
                const auto legal = env.decision().legal_actions.size();
                auto search = teacher::make_search(env.battle());
                const auto budget = legal == 1 ? teacher::forced_simulations : sims;
                std::vector<EncodedCombatState> enc;
                std::vector<float> values;
                while (search.simulations < budget) {
                    auto t = std::chrono::steady_clock::now();
                    const auto ids = search.requestBatch(teacher::value_net_batch, budget, 0, 0);
                    tree += seconds_since(t);
                    t = std::chrono::steady_clock::now();
                    enc.clear();
                    for (auto id : ids) enc.push_back(encode_state(search.pending.at(id).state));
                    encode += seconds_since(t);
                    t = std::chrono::steady_clock::now();
                    net.evaluate(enc, values);
                    eval += seconds_since(t);
                    t = std::chrono::steady_clock::now();
                    for (std::size_t i = 0; i < ids.size(); ++i) search.submit(ids[i], std::clamp<double>(values[i], 0, 2));
                    backup += seconds_since(t);
                }
                total += search.simulations;
                env.step(teacher::legal_index(env, legal, search, search.selectedAction()));
            }
        }
        std::cout << Json{{"sims", total}, {"tree", tree}, {"encode", encode}, {"eval", eval}, {"backup", backup}}.dump()
                  << std::endl;
        return 0;
    }
    for (auto seed = first; seed < first + count; ++seed) {
        auto env = scenarios::slime_boss(seed);
        const Json fight = {{"episode_id", seed}};
        std::vector<Json> rows;
        const auto start = std::chrono::steady_clock::now();
        const auto end = teacher::play_fight(env.battle(), fight, rows, run, false);
        if (trace)  // full rows (encodings + search output): must match exactly across behaviour-preserving changes
            for (const auto& r : rows) *trace << r.dump() << '\n';
        std::int64_t used = 0, decisions = 0;
        for (const auto& r : rows)
            if (r.at("row_kind") == "decision") { used += r.at("simulations_used").get<std::int64_t>(); ++decisions; }
        std::cout << Json{{"seed", seed}, {"seconds", seconds_since(start)}, {"decisions", decisions},
                          {"sims", used}, {"won", end.outcome == sts::Outcome::PLAYER_VICTORY},
                          {"hp", end.player.curHp}}.dump()
                  << std::endl;
    }
}
