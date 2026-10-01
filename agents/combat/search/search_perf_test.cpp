// The search speedups are exact: each fast path must give bit-for-bit what the code it replaced gave.
//   observationKey (tree node keys)  == the equality of publicObservation
//   encode_state                      == CombatEnvironment::decision().encoding without legal_actions
// States: random playouts and search leaves of Slime Boss fights.
#include "agents/combat/search/teacher_search.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/scenarios/slime_boss.hpp"

#include "combat/BattleContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using sts::search::PublicBeliefCombatSearch;
using namespace stsrl;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        std::exit(1);
    }
}

// Every decision state of random-move fights, plus the pending leaves of short searches along them.
std::vector<sts::BattleContext> sample_states() {
    std::vector<sts::BattleContext> states;
    std::mt19937_64 rng(7);
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        auto env = scenarios::slime_boss(seed);
        for (int step = 0; !env.done() && step < 200; ++step) {
            states.push_back(env.battle());
            const auto legal = env.decision().legal_actions.size();
            if (step % 7 == 0) {
                auto search = teacher::make_search(env.battle(), false, teacher::particles);
                while (search.simulations < 256) {
                    for (const auto id : search.requestBatch(64, 256, 0, 0)) {
                        states.push_back(search.pending.at(id).state);
                        search.submit(id, 0.5);
                    }
                }
            }
            env.step(std::uniform_int_distribution<std::size_t>{0, legal - 1}(rng));
        }
    }
    return states;
}

void test_observation_key(const std::vector<sts::BattleContext>& states) {
    std::unordered_map<std::uint64_t, std::uint64_t> fast_of, slow_of;
    for (const auto& s : states) {
        const auto slow = PublicBeliefCombatSearch::publicObservation(s);
        const auto fast = PublicBeliefCombatSearch::observationKey(s);
        const auto [a, new_slow] = fast_of.emplace(slow, fast);
        check(a->second == fast, "equal publicObservation, different observationKey");
        const auto [b, new_fast] = slow_of.emplace(fast, slow);
        check(b->second == slow, "equal observationKey, different publicObservation");
    }
    check(fast_of.size() > 100, "enough distinct observations");
}

void test_encode_state(const std::vector<sts::BattleContext>& states) {
    for (const auto& s : states) {
        if (s.outcome != sts::Outcome::UNDECIDED) continue;
        auto full = CombatEnvironment{s}.decision().encoding;
        full.legal_actions.clear();
        check(encode_state(s) == full, "encode_state differs from decision().encoding");
    }
}

// Merged edges: identical cards in two hand slots are one root edge; off, they are two.
void test_merge_identical_cards() {
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        auto env = scenarios::slime_boss(seed);
        const auto& hand = env.battle().cards;
        int strikes = 0;
        for (int i = 0; i < hand.cardsInHand; ++i) strikes += hand.hand[i].id == sts::CardId::STRIKE_RED;
        const auto count = [&](bool merge) {
            teacher::tweaks() = {};
            teacher::tweaks().merge_identical_cards = merge;
            return teacher::make_search(env.battle(), false, teacher::particles).root().edges.size();
        };
        const auto split = count(false), merged = count(true);
        teacher::tweaks() = {};
        check(merged <= split, "merging never adds edges");
        if (strikes >= 2) {
            // Adjacent duplicates were already one edge; non-adjacent ones are merged now.
            bool adjacent_only = true;
            for (int i = 0, last = -2; i < hand.cardsInHand; ++i)
                if (hand.hand[i].id == sts::CardId::STRIKE_RED) {
                    if (last >= 0 && i != last + 1) adjacent_only = false;
                    last = i;
                }
            if (!adjacent_only) check(merged < split, "non-adjacent identical Strikes merged");
        }
    }
}

void test_tweaks() {
    teacher::tweaks() = {};
    const teacher::Budget budget{teacher::simulations, teacher::particles};
    check(!teacher::settings({"value_net"}, false, budget).contains("stop_factor"), "no tweaks: none recorded");
    check(teacher::set_tweak("stop_factor", 0.25) && teacher::tweaks().stop_factor == 0.25, "stop_factor set");
    check(teacher::set_tweak("merge_identical_cards", true) && teacher::tweaks().merge_identical_cards, "merge set");
    const auto settings = teacher::settings({"value_net"}, false, budget);
    check(settings.at("stop_factor") == 0.25 && settings.at("merge_identical_cards") == true, "tweaks recorded");
    check(!teacher::set_tweak("simulations", 5), "other keys are not tweaks");
    for (const auto& bad : {nlohmann::json(0), nlohmann::json(1.5), nlohmann::json("x")}) {
        bool threw = false;
        try { teacher::set_tweak("stop_factor", bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid stop_factor rejected");
    }
    teacher::tweaks() = {};
}

}  // namespace

int main() {
    const auto states = sample_states();
    test_observation_key(states);
    test_encode_state(states);
    test_merge_identical_cards();
    test_tweaks();
    std::cout << "search_perf_test passed (" << states.size() << " states)\n";
}
