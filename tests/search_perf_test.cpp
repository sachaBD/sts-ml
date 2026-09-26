// The search speedups are exact: each fast path must give bit-for-bit what the code it replaced gave.
//   observationKey (tree node keys)  == the equality of publicObservation
//   encode_state                      == CombatEnvironment::decision().encoding without legal_actions
//   ValueNet (caches, AVX2 layers)    == the original ValueNet (tests/original_value_net.hpp), bit for bit
// States: random playouts and search leaves of Slime Boss fights.
#include "agents/teacher_leaves.hpp"
#include "combat/environment.hpp"
#include "models/value_net.hpp"
#include "scenarios/slime_boss.hpp"
#include "tests/original_value_net.hpp"

#include "combat/BattleContext.h"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
                auto search = teacher::make_search(env.battle());
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

// ---- ValueNet vs a plain forward pass ------------------------------------------------------------

struct Tensor {
    std::vector<std::uint32_t> shape;
    std::vector<float> data;
};
using Tensors = std::map<std::string, Tensor>;

// Random weights with the production shapes (width 64).
Tensors random_weights() {
    const std::vector<std::pair<std::string, std::vector<std::uint32_t>>> shapes = {
        {"input_state.weight", {64, 4}}, {"card_selection_task.weight", {32, 4}}, {"card_id.weight", {512, 8}},
        {"zone.weight", {5, 4}}, {"card_type.weight", {5, 3}}, {"target_type.weight", {4, 3}},
        {"card_mlp.0.weight", {64, 32}}, {"card_mlp.0.bias", {64}}, {"card_mlp.2.weight", {64, 64}},
        {"card_mlp.2.bias", {64}}, {"monster_id.weight", {128, 8}}, {"move.weight", {512, 8}},
        {"monster_mlp.0.weight", {64, 25}}, {"monster_mlp.0.bias", {64}}, {"monster_mlp.2.weight", {64, 64}},
        {"monster_mlp.2.bias", {64}}, {"interaction_mlp.0.weight", {64, 134}}, {"interaction_mlp.0.bias", {64}},
        {"interaction_mlp.2.weight", {64, 64}}, {"interaction_mlp.2.bias", {64}}, {"head.0.weight", {64, 442}},
        {"head.0.bias", {64}}, {"head.2.weight", {1, 64}}, {"head.2.bias", {1}}};
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 0.15f);
    Tensors tensors;
    for (const auto& [name, shape] : shapes) {
        Tensor t{shape, {}};
        std::size_t size = 1;
        for (auto d : shape) size *= d;
        for (std::size_t i = 0; i < size; ++i) t.data.push_back(normal(rng));
        tensors.emplace(name, std::move(t));
    }
    return tensors;
}

void write_weights(const Tensors& tensors, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    const auto u32 = [&](std::uint32_t x) { out.write(reinterpret_cast<const char*>(&x), sizeof x); };
    const std::string config = R"({"encoding_version": 3, "architecture": {"width": 64}})";
    out.write("STSVNET1", 8);
    u32(static_cast<std::uint32_t>(config.size()));
    out.write(config.data(), static_cast<std::streamsize>(config.size()));
    u32(static_cast<std::uint32_t>(tensors.size()));
    for (const auto& [name, t] : tensors) {
        u32(static_cast<std::uint32_t>(name.size()));
        out.write(name.data(), static_cast<std::streamsize>(name.size()));
        u32(static_cast<std::uint32_t>(t.shape.size()));
        for (auto d : t.shape) u32(d);
        out.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    }
}

void test_value_net(const std::vector<sts::BattleContext>& states) {
    const auto path = (std::filesystem::temp_directory_path() / "search_perf_test_value_weights.bin").string();
    write_weights(random_weights(), path);
    const ValueNet net{path};
    const original::OriginalValueNet reference{path};
    std::filesystem::remove(path);
    std::vector<EncodedCombatState> encoded;
    for (const auto& s : states)
        if (s.outcome == sts::Outcome::UNDECIDED) encoded.push_back(encode_state(s));
    std::vector<float> values, expected;
    reference.evaluate(encoded, expected);
    for (int pass = 0; pass < 2; ++pass) {  // second pass: every MLP output comes from the caches
        net.evaluate(encoded, values);
        for (std::size_t i = 0; i < encoded.size(); ++i)
            check(std::bit_cast<std::uint32_t>(values[i]) == std::bit_cast<std::uint32_t>(expected[i]),
                  "ValueNet differs from the original implementation (pass " + std::to_string(pass) + ")");
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
            return teacher::make_search(env.battle()).root().edges.size();
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

}  // namespace

int main() {
    const auto states = sample_states();
    test_observation_key(states);
    test_encode_state(states);
    test_value_net(states);
    test_merge_identical_cards();
    std::cout << "search_perf_test passed (" << states.size() << " states)\n";
}
