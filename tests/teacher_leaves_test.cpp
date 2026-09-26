// Teacher leaf strategies (agents/teacher_leaves.hpp) and the play_fight exploration toggle.
#include "agents/teacher_search.hpp"
#include "scenarios/jaw_worm.hpp"

#include "combat/BattleContext.h"
#include "constants/Cards.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using Json = nlohmann::json;
namespace teacher = stsrl::teacher;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        std::exit(1);
    }
}

template <class F>
bool throws(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

sts::BattleContext lethal_jaw_worm_state() {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, 17, 1};
    game.floorNum = 1;
    game.curRoom = sts::Room::MONSTER;
    sts::BattleContext battle;
    battle.init(game, sts::MonsterEncounter::JAW_WORM);
    battle.cards.cardsInHand = 1;
    battle.cards.hand[0] = sts::CardInstance{sts::CardId::STRIKE_RED};
    battle.monsters.arr[0].curHp = 1;
    battle.monsters.arr[0].block = 0;
    return battle;
}

// Records every evaluated leaf; value from the first monster's HP (deterministic, state-dependent).
struct Recorder {
    std::vector<int> turns;
    int undecided = 0;
    teacher::LeafEvaluator evaluator(float constant = -1) {
        return [this, constant](const std::vector<const sts::BattleContext*>& leaves, std::vector<float>& values) {
            for (const auto* leaf : leaves) {
                turns.push_back(leaf->turn);
                undecided += leaf->outcome == sts::Outcome::UNDECIDED;
                values.push_back(constant >= 0 ? constant : 1.0f - leaf->monsters.arr[0].curHp / 100.0f);
            }
        };
    }
};

struct SearchResult {
    std::int64_t used;
    std::vector<std::pair<std::int64_t, double>> edges;
};

SearchResult leaf_search(const sts::BattleContext& root, Recorder& recorder, std::int64_t budget, int turns,
                         int steps, bool defaults = false) {
    stsrl::CombatEnvironment env{root};
    const auto legal = env.decision().legal_actions.size();
    auto search = teacher::make_search(root);
    const auto used = defaults ? teacher::run_leaf_search(search, recorder.evaluator(), budget, legal)
                               : teacher::run_leaf_search(search, recorder.evaluator(), budget, legal, turns, steps);
    SearchResult result{used, {}};
    for (const auto& e : search.root().edges) result.edges.emplace_back(e.visits, e.valueSum);
    return result;
}

void test_validation() {
    using teacher::Leaf;
    check(!throws([] { teacher::validate(Leaf{"guided_rollout"}, false); }), "guided_rollout valid");
    check(!throws([] { teacher::validate(Leaf{"value_net"}, true); }), "value_net valid");
    check(!throws([] { teacher::validate(Leaf{"hybrid", 1, 16}, true); }), "hybrid valid");
    check(throws([] { teacher::validate(Leaf{"guided_rollout"}, true); }), "guided_rollout with net");
    check(throws([] { teacher::validate(Leaf{"guided_rollout", 1, 16}, false); }), "guided_rollout with bounds");
    check(throws([] { teacher::validate(Leaf{"value_net"}, false); }), "value_net without net");
    check(throws([] { teacher::validate(Leaf{"value_net", 1, 16}, true); }), "value_net with bounds");
    check(throws([] { teacher::validate(Leaf{"hybrid", 0, 0}, true); }), "hybrid zero bounds");
    check(throws([] { teacher::validate(Leaf{"hybrid", 1, 0}, true); }), "hybrid zero steps");
    check(throws([] { teacher::validate(Leaf{"hybrid", 0, 16}, true); }), "hybrid zero turns");
    check(throws([] { teacher::validate(Leaf{"hybrid", 1, 16}, false); }), "hybrid without net");
    check(throws([] { teacher::validate(Leaf{"mixed"}, true); }), "unknown leaf");
    check(throws([] { teacher::leaf_search(Leaf{"value_net"}, nullptr); }), "leaf_search validates");
    Recorder recorder;
    check(throws([&] { leaf_search(stsrl::scenarios::jaw_worm(1).battle(), recorder, 10, -1, 0); }),
          "negative rollout bound");
}

void test_settings() {
    // Bootstrap's recorded teacher settings are unchanged by the split.
    const Json guided = {{"leaf", "guided_rollout"}, {"particles", 8}, {"simulations", 15000}, {"early_stop", true},
                         {"forced_simulations", 500}, {"max_actions", 512}, {"child_min_visits", 50},
                         {"random_window", 24}, {"chunk", 500}, {"oracle", false}};
    check(teacher::settings("guided_rollout") == guided, "guided_rollout settings");
    auto value = guided;
    value["leaf"] = "value_net";
    value.erase("chunk");
    value["batch"] = 64;
    check(teacher::settings("value_net") == value, "value_net settings");
    auto hybrid = value;
    hybrid["leaf"] = "hybrid";
    hybrid["rollout_turns"] = 1;
    hybrid["rollout_steps"] = 16;
    check(teacher::settings(teacher::Leaf{"hybrid", 1, 16}) == hybrid, "hybrid settings");
}

void test_leaf_search() {
    const auto root = stsrl::scenarios::jaw_worm(3).battle();
    // Zero bounds are the immediate path: same search as the default arguments.
    Recorder a, b;
    const auto defaults = leaf_search(root, a, 300, 0, 0, true);
    const auto zero = leaf_search(root, b, 300, 0, 0);
    check(defaults.used == zero.used && defaults.edges == zero.edges && a.turns == b.turns, "zero bounds == default");
    check(!a.turns.empty(), "immediate evaluates leaves");
    check(std::ranges::any_of(a.turns, [&](int t) { return t == root.turn; }), "immediate leaves at the root turn");

    // Positive bounds: each leaf is rolled out at least one turn before it is evaluated.
    Recorder h;
    const auto hybrid = leaf_search(root, h, 300, 1, 16);
    check(hybrid.used > 0 && hybrid.used <= 300, "hybrid budget");
    check(!h.turns.empty() && h.undecided == static_cast<int>(h.turns.size()), "hybrid leaves undecided");
    check(std::ranges::all_of(h.turns, [&](int t) { return t >= root.turn + 1; }), "hybrid leaves rolled a turn");

    // Terminal handling: evaluator says 0 everywhere; only terminal wins can give value.
    for (const auto& [turns, steps] : {std::pair{0, 0}, std::pair{1, 16}}) {
        const auto lethal = lethal_jaw_worm_state();
        stsrl::CombatEnvironment env{lethal};
        const auto legal = env.decision().legal_actions.size();
        check(legal > 1, "lethal state has a choice");
        auto search = teacher::make_search(lethal);
        Recorder r;
        const auto used = teacher::run_leaf_search(search, r.evaluator(0), 200, legal, turns, steps);
        check(used == search.simulations && used > 0 && used <= 200, "terminal budget");
        check(r.undecided == static_cast<int>(r.turns.size()), "terminal leaves never evaluated");
        check(std::ranges::any_of(search.root().edges, [](const auto& e) { return e.valueSum > 0; }),
              "terminal win backed up");
    }
}

std::uint64_t episode_with_random_at_zero() {
    for (std::uint64_t id = 0;; ++id) {
        std::mt19937_64 rng(id ^ 0xe9510ULL);
        if (std::uniform_int_distribution<int>{0, teacher::random_window - 1}(rng) == 0) return id;
    }
}

void test_exploration() {
    const auto cheap = teacher::SearchFn{[](auto& search, std::size_t) { search.search(20); return std::int64_t{20}; }};
    const Json fight = {{"episode_id", episode_with_random_at_zero()}};
    const auto battle = stsrl::scenarios::jaw_worm(5).battle();
    const auto play = [&](int mode) {
        std::vector<Json> rows;
        if (mode < 0) teacher::play_fight(battle, fight, rows, cheap);
        else teacher::play_fight(battle, fight, rows, cheap, mode == 1);
        return rows;
    };
    const auto defaults = play(-1), on = play(1), off = play(0);
    check(defaults == on, "default is random_move = true");
    const auto random_rows = [](const std::vector<Json>& rows) {
        return std::ranges::count_if(rows, [](const Json& r) { return r.at("was_random").get<bool>(); });
    };
    check(random_rows(on) == 1, "random_move = true: one random move");
    const auto first = std::ranges::find_if(on, [](const Json& r) { return r.at("row_kind") == "decision"; });
    check(first->at("decision_index") == 0 && first->at("was_random") == true, "random move at decision 0");
    check(random_rows(off) == 0, "random_move = false: no random move");
    check(!off.empty(), "random_move = false plays the fight");
}

}  // namespace

int main() {
    test_validation();
    test_settings();
    test_leaf_search();
    test_exploration();
    std::cout << "teacher_leaves_test: ok\n";
}
