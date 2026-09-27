// One fight of a combat_v3 data run, replayed with a teacher leaf strategy. The act 1 run is rebuilt
// up to that fight: SimpleAgent out of combat (as in bootstrap), the earlier fights by their stored
// chosen actions. Then the teacher plays the fight and the worker stops.
//   value_play_worker REQUEST.json OUTPUT_DIR [WEIGHTS]     (apps/common/worker.hpp; no WEIGHTS: no value net)
//   REQUEST.json: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight],
//                teacher: apps/common/teacher_request.hpp}
//   oracle: search the true state (perfect RNG foresight, teacher_leaves.hpp make_search).
//   Invalid combinations (e.g. hybrid without bounds, guided_rollout with weights) fail.
// Output: OUTPUT_DIR/result.msgpack = {episode_id, teacher, rows} (combat_v3 rows of that fight);
// teacher = teacher::settings(leaf, oracle, budget) + random_move.
#include "agents/teacher_search.hpp"
#include "apps/common/fight_replay.hpp"
#include "apps/common/teacher_request.hpp"
#include "apps/common/worker.hpp"

#include <vector>

namespace {
using Json = nlohmann::json;

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto teacher = stsrl::teacher::parse_request(input.at("teacher"));
    const auto search = stsrl::teacher::leaf_search(teacher.leaf, net, teacher.budget);  // validates leaf vs net
    std::vector<Json> rows;
    const auto fight = stsrl::replay::to_fight(input, [&](const sts::BattleContext& start, const Json& columns) {
        return stsrl::teacher::play_fight(start, columns, rows, search, teacher.random_move, teacher.oracle,
                                            teacher.budget.particles, false);
    });
    const auto episode = fight.at("episode_id");
    auto settings = stsrl::teacher::settings(teacher.leaf, teacher.oracle, teacher.budget);
    settings["random_move"] = teacher.random_move;
    return {{"episode_id", episode}, {"teacher", settings}, {"rows", rows}};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "value_play_worker", play); }
