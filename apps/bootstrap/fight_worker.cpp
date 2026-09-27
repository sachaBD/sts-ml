// One seeded Ironclad act 1 (combat_v3, see python/sts_combat_rl/schemas/combat_v3.py). Runs whose
// act 1 boss isn't selected stop at game creation. SimpleAgent plays everything out of combat;
// the teacher (agents/teacher_search.hpp) plays and records every combat until death or the boss is beaten.
//   bootstrap_fight_worker REQUEST.json OUTPUT_DIR     (apps/common/worker.hpp)
//   REQUEST.json: {seed, ascension (0..20), oracle, bosses, teacher, simulations};
//   bosses: selected act 1 boss names; teacher: opt-in search tweaks {stop_factor, merge_identical_cards} (may be empty);
//   simulations: {easy, hard, elite, event, boss} search budgets, all required.
// Output: OUTPUT_DIR/result.msgpack = {seed, status, floor, fights, teacher, rows}.
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

Json play_run(std::uint64_t seed, int ascension, bool oracle, const Json& bosses, const Json& options,
              const Json& simulations) {
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
    auto teacher = stsrl::teacher::settings("guided_rollout", oracle);
    teacher["simulations_by_category"] = simulations;
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    if (!bosses.is_array() || bosses.empty()) throw std::invalid_argument{"bosses must be a nonempty array"};
    bool selected = false;
    for (const auto& name : bosses) {
        if (!name.is_string()) throw std::invalid_argument{"boss names must be strings"};
        selected |= game.boss == boss_from_name(name.get<std::string>());
    }
    if (!selected)  // act 1 boss is fixed at game creation
        return {{"seed", seed}, {"status", "other_boss"}, {"floor", 0}, {"fights", 0}, {"teacher", teacher},
                {"rows", Json::array()}};
    std::vector<Json> rows;
    const auto run = stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json& fight) {
        const auto category = fight.at("category").get<std::string>();
        const auto search = stsrl::teacher::guided_rollout_search(
            simulations.at(category).get<std::int64_t>());
        return stsrl::teacher::play_fight(start, fight, rows, search, true, oracle);
    });
    return {{"seed", seed}, {"status", run.status}, {"floor", run.floor}, {"fights", run.fights}, {"teacher", teacher},
            {"rows", rows}};
}

Json play(const Json& request, const stsrl::ValueNet*) {
    const auto ascension = request.at("ascension").get<int>();
    if (ascension < 0 || ascension > 20) throw std::invalid_argument{"ascension must be in [0, 20]"};
    return play_run(request.at("seed").get<std::uint64_t>(), ascension, request.at("oracle").get<bool>(),
                    request.at("bosses"), request.at("teacher"), request.at("simulations"));
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "bootstrap_fight_worker", play); }
