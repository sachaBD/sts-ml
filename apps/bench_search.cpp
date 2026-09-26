// Search performance benchmark (data-gen teacher: value-net leaves, 20k simulations, 8 particles, no random move).
//   bench_search play WEIGHTS FIRST COUNT [SIMS]   -> one JSON line per fight (time, sims, outcome)
// Slime Boss scenario, one core.
#include "agents/teacher_search.hpp"
#include "models/value_net.hpp"
#include "scenarios/slime_boss.hpp"

#include <chrono>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <iostream>
#include <string>

using namespace stsrl;
using Json = nlohmann::json;

namespace sts::search { extern unsigned long long g_prof[8]; }  // PBCS_PROFILE builds

namespace {
// Thread CPU time: the machine is shared, so wall time also counts time spent waiting for a core.
double cpu_now() {
    timespec t{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return static_cast<double>(t.tv_sec) + 1e-9 * static_cast<double>(t.tv_nsec);
}
double seconds_since(double start) { return cpu_now() - start; }
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
    const auto run = std::getenv("GUIDED") ? teacher::guided_rollout_search(sims) : teacher::value_net_search(net, sims);
    std::unique_ptr<std::ofstream> trace;
    if (const char* path = std::getenv("TRACE")) trace = std::make_unique<std::ofstream>(path);
    if (mode == "regret") {
        // Decision quality along the baseline teacher's trajectory. Per state: a reference search (merged
        // edges, REF_SIMS simulations, no early stop) gives Q_ref per move (identityActionKey); then each arm
        // of ARMS (comma list; an arm is '+'-joined tokens: base, reseed, merge, stop=F, sims=N, reuse)
        // searches the state. One JSON line per state; analysis: regret = max Q_ref - Q_ref(chosen).
        struct Arm {
            std::string name;
            teacher::SearchTweaks tweaks;
            std::int64_t sims = 20000;
            bool reseed = false, reuse = false;
            std::optional<sts::search::PublicBeliefCombatSearch> tree;
        };
        std::vector<Arm> arms;
        {
            std::stringstream list(std::getenv("ARMS") ? std::getenv("ARMS") : "base");
            for (std::string spec; std::getline(list, spec, ',');) {
                Arm arm{spec};
                arm.sims = sims;
                std::stringstream tokens(spec);
                for (std::string t; std::getline(tokens, t, '+');) {
                    if (t == "merge") arm.tweaks.merge_identical_cards = true;
                    else if (t == "dag") arm.tweaks.transpositions = true;
                    else if (t == "reseed") arm.reseed = true;
                    else if (t == "reuse") arm.reuse = true;
                    else if (t.starts_with("stop=")) arm.tweaks.stop_factor = std::stod(t.substr(5));
                    else if (t.starts_with("sims=")) arm.sims = std::stoll(t.substr(5));
                    else if (t != "base") throw std::invalid_argument{"unknown arm token " + t};
                }
                arms.push_back(std::move(arm));
            }
        }
        const bool ref_merge = !std::getenv("REF_MERGE") || std::string{std::getenv("REF_MERGE")} != "0";
        const std::int64_t ref_sims = std::getenv("REF_SIMS") ? std::stoll(std::getenv("REF_SIMS")) : 100000;
        const auto key_of = [](const sts::BattleContext& state, std::uint32_t bits) {
            return std::to_string(sts::search::PublicBeliefCombatSearch::identityActionKey(state, sts::search::Action{bits}));
        };
        for (auto seed = first; seed < first + count; ++seed) {
            auto env = scenarios::slime_boss(seed);
            for (auto& arm : arms) arm.tree.reset();
            for (int index = 0; !env.done(); ++index) {
                const auto legal = env.decision().legal_actions.size();
                Json line = {{"seed", seed}, {"decision", index}, {"legal", legal}};
                std::size_t played = 0;
                // Arms first (their trees must see every state), then the reference (multi-move states only).
                Json out = Json::object();
                for (std::size_t a = 0; a < arms.size(); ++a) {
                    auto& arm = arms[a];
                    teacher::tweaks() = arm.tweaks;
                    const auto arm_run = teacher::value_net_search(net, arm.sims);
                    const auto t = cpu_now();
                    if (!arm.reuse || !arm.tree) arm.tree.emplace(teacher::make_search(env.battle()));
                    if (arm.reseed) {
                        arm.tree->random.seed(seed * 7919 + index);
                        arm.tree->rollout.randGen.seed(seed * 7919 + index);
                    }
                    const auto d = teacher::search_decision(env, legal, arm_run, *arm.tree);
                    out[arm.name] = {{"move", key_of(env.battle(), env.action_bits(d.chosen))}, {"sims", d.used},
                                     {"retained", d.retained}, {"s", seconds_since(t)}, {"value", d.value}};
                    if (a == 0) played = d.chosen;
                }
                line["arms"] = out;
                if (legal > 1) {
                    teacher::tweaks() = {};
                    teacher::tweaks().merge_identical_cards = ref_merge;
                    teacher::tweaks().stop_factor = 1e30;
                    auto ref = teacher::make_search(env.battle());
                    teacher::run_value_net_search(ref, net, ref_sims, legal);
                    std::map<std::string, std::pair<std::int64_t, double>> moves;  // duplicates (unmerged) pooled
                    for (const auto& e : ref.root().edges) {
                        auto& m = moves[key_of(ref.particles.front(), e.action.bits)];
                        m.first += e.visits; m.second += e.valueSum;
                    }
                    Json q = Json::object();
                    for (const auto& [k, m] : moves) q[k] = {{"n", m.first}, {"q", m.first ? m.second / m.first : 0.0}};
                    line["ref"] = q;
                }
                std::cout << line.dump() << std::endl;
                const auto before = env.battle();
                const auto bits = env.action_bits(played);
                env.step(played);
                for (auto& arm : arms)
                    if (arm.reuse && !env.done()) {
                        teacher::tweaks() = arm.tweaks;
                        teacher::rebase_search(*arm.tree, before, bits, env.battle());
                    }
            }
        }
        return 0;
    }
    if (mode == "paired") {
        // Along the baseline teacher's trajectory, each state is also searched by the variant (VARIANT env:
        // comma list of merge) and, as a noise control, by the baseline with a reseeded search RNG.
        const std::string variant = std::getenv("VARIANT") ? std::getenv("VARIANT") : "";
        const auto set_variant = [&](bool on) {
            teacher::tweaks() = {};
            if (on && variant.find("merge") != std::string::npos) teacher::tweaks().merge_identical_cards = true;
        };
        const auto move_key = [](const CombatEnvironment& env, std::size_t i) {
            return sts::search::PublicBeliefCombatSearch::identityActionKey(env.battle(),
                                                                            sts::search::Action{env.action_bits(i)});
        };
        for (auto seed = first; seed < first + count; ++seed) {
            auto env = scenarios::slime_boss(seed);
            for (int index = 0; !env.done(); ++index) {
                const auto legal = env.decision().legal_actions.size();
                set_variant(false);
                auto t = cpu_now();
                const auto base = teacher::search_decision(env, legal, run);
                const double base_s = seconds_since(t);
                set_variant(true);
                t = cpu_now();
                const auto var = teacher::search_decision(env, legal, run);
                const double var_s = seconds_since(t);
                set_variant(false);
                auto other = teacher::make_search(env.battle());
                other.random.seed(seed * 7919 + index);
                other.rollout.randGen.seed(seed * 7919 + index);
                const auto control = teacher::search_decision(env, legal, run, other);
                std::cout << Json{{"seed", seed}, {"decision", index}, {"legal", legal},
                                  {"base_s", base_s}, {"base_sims", base.used}, {"base_value", base.value},
                                  {"var_s", var_s}, {"var_sims", var.used}, {"var_value", var.value},
                                  {"same", move_key(env, base.chosen) == move_key(env, var.chosen)},
                                  {"control_same", move_key(env, base.chosen) == move_key(env, control.chosen)}}.dump()
                          << std::endl;
                env.step(base.chosen);
            }
        }
        return 0;
    }
    if (mode == "evalbench") {
        // Leaf states of the first decisions' searches, then timed encode / evaluate passes over them.
        std::vector<sts::BattleContext> leaves;
        auto env = scenarios::slime_boss(first);
        for (int d = 0; d < 6 && !env.done(); ++d) {
            auto search = teacher::make_search(env.battle());
            while (search.simulations < 5000) {
                const auto ids = search.requestBatch(teacher::value_net_batch, 5000, 0, 0);
                for (auto id : ids) { leaves.push_back(search.pending.at(id).state); search.submit(id, 0.5); }
            }
            env.step(teacher::legal_index(env, env.decision().legal_actions.size(), search, search.selectedAction()));
        }
        const int reps = count > 0 ? static_cast<int>(count) : 1;
        double encode = 0, eval = 0, sum = 0;
        std::vector<EncodedCombatState> enc;
        std::vector<float> values;
        for (int r = 0; r < reps; ++r) {
            auto t = cpu_now();
            enc.clear();
            for (const auto& l : leaves) enc.push_back(encode_state(l));
            encode += seconds_since(t);
            t = cpu_now();
            net.evaluate(enc, values);
            eval += seconds_since(t);
        }
        for (float v : values) sum += v;
        const double n = static_cast<double>(leaves.size()) * reps;
        std::cout << Json{{"leaves", leaves.size()}, {"encode_us", encode / n * 1e6}, {"eval_us", eval / n * 1e6},
                          {"checksum", sum}}.dump() << std::endl;
        return 0;
    }
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
                    auto t = cpu_now();
                    const auto ids = search.requestBatch(teacher::value_net_batch, budget, 0, 0);
                    tree += seconds_since(t);
                    t = cpu_now();
                    enc.clear();
                    for (auto id : ids) enc.push_back(encode_state(search.pending.at(id).state));
                    encode += seconds_since(t);
                    t = cpu_now();
                    net.evaluate(enc, values);
                    eval += seconds_since(t);
                    t = cpu_now();
                    for (std::size_t i = 0; i < ids.size(); ++i) search.submit(ids[i], std::clamp<double>(values[i], 0, 2));
                    backup += seconds_since(t);
                }
                total += search.simulations;
                env.step(teacher::legal_index(env, legal, search, search.selectedAction()));
            }
        }
        std::cout << Json{{"sims", total}, {"tree", tree}, {"encode", encode}, {"eval", eval}, {"backup", backup},
                          {"tree_cycles", std::vector<unsigned long long>(sts::search::g_prof, sts::search::g_prof + 6)}}.dump()
                  << std::endl;
        return 0;
    }
    for (auto seed = first; seed < first + count; ++seed) {
        auto env = scenarios::slime_boss(seed);
        const Json fight = {{"episode_id", seed}};
        std::vector<Json> rows;
        const auto start = cpu_now();
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
