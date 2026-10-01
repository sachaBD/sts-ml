// One seeded Ironclad act 1 (combat_v3, see python/sts_combat_rl/schemas/combat_v3.py). Runs whose
// act 1 boss isn't selected stop at game creation. SimpleAgent plays everything out of combat;
// the teacher (agents/teacher_search.hpp) plays and records every combat until death or the boss is beaten.
//   bootstrap_fight_worker REQUEST.json OUTPUT_DIR [WEIGHTS]     (apps/common/worker.hpp)
//   REQUEST.json: {seed, ascension (0..20), oracle, bosses, teacher, simulations, leaf, random_move};
//   bosses: selected act 1 boss names; teacher: opt-in search tweaks {stop_factor, merge_identical_cards} (may be empty);
//   simulations: {easy, hard, elite, event, boss} search budgets, all required;
//   leaf: {kind (guided_rollout | value_net | hybrid), rollout_turns, rollout_steps} (teacher_leaves.hpp Leaf;
//   net leaves need WEIGHTS); random_move: one random move per fight (teacher_search.hpp play_fight).
// Output: OUTPUT_DIR/result.msgpack = {seed, boss, status, floor, fights, final_hp, teacher, rows}.
#include "agents/teacher_search.hpp"
#include "apps/common/worker.hpp"
#include "constants/CharacterClasses.h"
#include "scenarios/act1_run.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

sts::MonsterEncounter boss_from_name(const std::string& name) {
    if (name == "slime_boss") return sts::MonsterEncounter::SLIME_BOSS;
    if (name == "the_guardian") return sts::MonsterEncounter::THE_GUARDIAN;
    if (name == "hexaghost") return sts::MonsterEncounter::HEXAGHOST;
    throw std::invalid_argument{"unknown act 1 boss: " + name};
}

std::string boss_name(sts::MonsterEncounter boss) {
    for (const auto* name : {"slime_boss", "the_guardian", "hexaghost"})
        if (boss_from_name(name) == boss) return name;
    throw std::invalid_argument{"not an act 1 boss"};
}

Json play_run(std::uint64_t seed, int ascension, bool oracle, const Json& bosses, const Json& options,
              const Json& simulations, const stsrl::teacher::Leaf& leaf, bool random_move,
              const stsrl::ValueNet* net) {
    if (!options.is_object()) throw std::invalid_argument{"teacher must be an object"};
    for (const auto& [key, value] : options.items())
        if (!stsrl::teacher::set_tweak(key, value)) throw std::invalid_argument{"unknown teacher setting: " + key};
    if (!simulations.is_object()) throw std::invalid_argument{"simulations must be an object"};
    if (simulations.size() != 5) throw std::invalid_argument{"simulations needs exactly easy, hard, elite, event, boss"};
    for (const auto* key : {"easy", "hard", "elite", "event", "boss"}) {
        const auto& value = simulations.at(key);
        if (!value.is_number_integer() || value.is_boolean() || value.get<std::int64_t>() < 1)
            throw std::invalid_argument{std::string{"simulations for "} + key + " must be a positive integer"};
    }
    // The bootstrap teacher: `leaf` leaves, teacher::particles, simulations by fight category.
    namespace teacher_ = stsrl::teacher;
    teacher_::validate(leaf, net != nullptr);
    auto teacher = teacher_::settings(leaf, oracle, {teacher_::simulations, teacher_::particles});
    teacher.erase("simulations");
    teacher["simulations_by_category"] = simulations;
    teacher["random_move"] = random_move;
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    if (!bosses.is_array() || bosses.empty()) throw std::invalid_argument{"bosses must be a nonempty array"};
    bool selected = false;
    for (const auto& name : bosses) {
        if (!name.is_string()) throw std::invalid_argument{"boss names must be strings"};
        selected |= game.boss == boss_from_name(name.get<std::string>());
    }
    if (!selected)  // act 1 boss is fixed at game creation
        return {{"seed", seed}, {"boss", boss_name(game.boss)}, {"status", "other_boss"}, {"floor", 0},
                {"fights", 0}, {"final_hp", game.curHp}, {"teacher", teacher}, {"rows", Json::array()}};
    std::vector<Json> rows;
    const auto run = stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json& fight) {
        const auto category = fight.at("category").get<std::string>();
        const auto search = teacher_::leaf_search(
            leaf, net, {simulations.at(category).get<std::int64_t>(), teacher_::particles});
        return teacher_::play_fight(start, fight, rows, search, random_move, oracle, teacher_::particles, false);
    });
    return {{"seed", seed}, {"boss", boss_name(game.boss)}, {"status", run.status}, {"floor", run.floor},
            {"fights", run.fights}, {"final_hp", game.curHp}, {"teacher", teacher}, {"rows", rows}};
}

stsrl::teacher::Leaf parse_leaf(const Json& leaf) {
    if (!leaf.is_object()) throw std::invalid_argument{"leaf must be an object"};
    for (const auto& [key, value] : leaf.items())
        if (key != "kind" && key != "rollout_turns" && key != "rollout_steps")
            throw std::invalid_argument{"unknown leaf setting: " + key};
    return {leaf.at("kind").get<std::string>(), leaf.value("rollout_turns", 0), leaf.value("rollout_steps", 0)};
}

Json play(const Json& request, const stsrl::ValueNet* net) {
    const auto ascension = request.at("ascension").get<int>();
    if (ascension < 0 || ascension > 20) throw std::invalid_argument{"ascension must be in [0, 20]"};
    return play_run(request.at("seed").get<std::uint64_t>(), ascension, request.at("oracle").get<bool>(),
                    request.at("bosses"), request.at("teacher"), request.at("simulations"),
                    parse_leaf(request.at("leaf")), request.at("random_move").get<bool>(), net);
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "bootstrap_fight_worker", play); }
