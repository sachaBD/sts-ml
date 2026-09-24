// One seeded Ironclad act 1 (combat_v3, see python/sts_combat_rl/schemas/combat_v3.py). Runs whose
// act 1 boss isn't Slime Boss stop at game creation. SimpleAgent plays everything out of combat;
// the teacher (agents/teacher_search.hpp) plays and records every combat until death or the boss is beaten.
// ORACLE (optional, 0/1): teacher searches the true state (teacher_leaves.hpp make_search).
// Output: OUTPUT_DIR/fight.msgpack = {seed, status, floor, fights, teacher, rows}.
#include "agents/teacher_search.hpp"
#include "apps/worker_io.hpp"
#include "constants/CharacterClasses.h"
#include "scenarios/act1_run.hpp"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;

std::uint64_t parse_seed(std::string_view text) {
    std::uint64_t seed{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument{"SEED must be an unsigned integer"};
    return seed;
}

int parse_ascension(std::string_view text) {
    int ascension{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), ascension);
    if (error != std::errc{} || end != text.data() + text.size() || ascension < 0 || ascension > 20)
        throw std::invalid_argument{"ASCENSION must be an integer in [0, 20]"};
    return ascension;
}

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

}  // namespace

int main(int argc, char* argv[]) {
    const std::string_view oracle_arg = argc == 5 ? argv[4] : "0";
    if ((argc != 4 && argc != 5) || (oracle_arg != "0" && oracle_arg != "1")) {
        std::cerr << "usage: bootstrap_fight_worker SEED ASCENSION OUTPUT_DIR [ORACLE=0|1]\n";
        return 2;
    }
    try {
        stsrl::worker::write_result(argv[3], play_run(parse_seed(argv[1]), parse_ascension(argv[2]), oracle_arg == "1"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fight worker (seed " << argv[1] << "): " << error.what() << '\n';
        return 1;
    }
}
