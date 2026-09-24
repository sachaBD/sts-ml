// One seeded Ironclad act 1 (combat_v3, see python/sts_combat_rl/schemas/combat_v3.py). Runs whose
// act 1 boss isn't Slime Boss stop at game creation. SimpleAgent plays everything out of combat;
// the teacher (agents/teacher_search.hpp) plays and records every combat until death or the boss is beaten.
//   bootstrap_fight_worker REQUEST.json OUTPUT_DIR     (apps/common/worker.hpp)
//   REQUEST.json: {seed, ascension (0..20), oracle}; oracle: the teacher searches the true state (teacher_leaves.hpp make_search).
// Output: OUTPUT_DIR/result.msgpack = {seed, status, floor, fights, teacher, rows}.
#include "agents/teacher_search.hpp"
#include "apps/common/worker.hpp"
#include "constants/CharacterClasses.h"
#include "scenarios/act1_run.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::json;

Json play_run(std::uint64_t seed, int ascension, bool oracle) {
    const auto teacher = stsrl::teacher::settings("guided_rollout", oracle);
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    if (game.boss != sts::MonsterEncounter::SLIME_BOSS)  // act 1 boss is fixed at game creation
        return {{"seed", seed}, {"status", "other_boss"}, {"floor", 0}, {"fights", 0}, {"teacher", teacher},
                {"rows", Json::array()}};
    std::vector<Json> rows;
    const auto search = stsrl::teacher::guided_rollout_search();
    const auto run = stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json& fight) {
        return stsrl::teacher::play_fight(start, fight, rows, search, true, oracle);
    });
    return {{"seed", seed}, {"status", run.status}, {"floor", run.floor}, {"fights", run.fights}, {"teacher", teacher},
            {"rows", rows}};
}

Json play(const Json& request, const stsrl::ValueNet*) {
    const auto ascension = request.at("ascension").get<int>();
    if (ascension < 0 || ascension > 20) throw std::invalid_argument{"ascension must be in [0, 20]"};
    return play_run(request.at("seed").get<std::uint64_t>(), ascension, request.at("oracle").get<bool>());
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "bootstrap_fight_worker", play); }
