// Tree reuse benchmark: Slime Boss scenario, guided-rollout teacher (15k budget), one JSON line per decision.
//   bench_reuse first count
// Plays the fresh-search teacher. At every decision the same state is also searched with the reused tree
// (rebased along the played moves) and, as a noise control, by a fresh search with a reseeded search RNG.
#include "agents/teacher_search.hpp"
#include "scenarios/slime_boss.hpp"

#include <chrono>
#include <iostream>
#include <string>

using namespace stsrl;
using Json = nlohmann::json;

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void paired(std::uint64_t seed) {
    const auto run = teacher::guided_rollout_search(teacher::simulations);
    auto env = scenarios::slime_boss(seed);
    auto start = std::chrono::steady_clock::now();
    auto tree = teacher::make_search(env.battle());
    double setup = seconds_since(start);  // make_search / rebase time, charged to the reused search
    for (int index = 0; !env.done(); ++index) {
        const auto legal = env.decision().legal_actions.size();
        start = std::chrono::steady_clock::now();
        const auto fresh = teacher::search_decision(env, legal, run);
        const double fresh_seconds = seconds_since(start);
        start = std::chrono::steady_clock::now();
        const auto reused = teacher::search_decision(env, legal, run, tree);
        const double reused_seconds = setup + seconds_since(start);
        auto other = teacher::make_search(env.battle());
        other.random.seed(seed * 7919 + index);
        other.rollout.randGen.seed(seed * 7919 + index);
        const auto control = teacher::search_decision(env, legal, run, other);
        std::cout << Json{{"seed", seed}, {"decision", index}, {"legal", legal},
                          {"fresh_seconds", fresh_seconds}, {"fresh_sims", fresh.used},
                          {"reused_seconds", reused_seconds}, {"reused_sims", reused.used},
                          {"retained", reused.retained}, {"same_move", fresh.chosen == reused.chosen},
                          {"control_same_move", fresh.chosen == control.chosen}}.dump()
                  << std::endl;
        const auto before = env.battle();
        const auto bits = env.action_bits(fresh.chosen);
        env.step(fresh.chosen);
        start = std::chrono::steady_clock::now();
        if (!env.done()) teacher::rebase_search(tree, before, bits, env.battle());
        setup = seconds_since(start);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: bench_reuse first count\n";
        return 2;
    }
    const auto first = std::stoull(argv[1]), count = std::stoull(argv[2]);
    for (auto seed = first; seed < first + count; ++seed) paired(seed);
}
