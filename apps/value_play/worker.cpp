// One fight of a combat_v3 data run, replayed with a teacher leaf strategy. The act 1 run is rebuilt
// up to that fight: SimpleAgent out of combat (as in bootstrap), the earlier fights by their stored
// chosen actions. Then the teacher plays the fight and the worker stops.
//   value_play_worker REQUEST.json OUTPUT_DIR [WEIGHTS]     (apps/common/worker.hpp; no WEIGHTS: no value net)
//   REQUEST.json: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight],
//                teacher (optional): {leaf, rollout_turns, rollout_steps, random_move, oracle}}
//   oracle: search the true state (perfect RNG foresight, teacher_leaves.hpp make_search).
//   teacher defaults: leaf value_net, rollout bounds 0, random_move true (the original value_play).
//   Invalid combinations (e.g. hybrid without bounds, guided_rollout with weights) fail.
// Output: OUTPUT_DIR/result.msgpack = {episode_id, teacher, rows} (combat_v3 rows of that fight);
// teacher = teacher::settings(leaf) + random_move.
#include "agents/teacher_search.hpp"
#include "apps/common/fight_replay.hpp"
#include "apps/common/worker.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

struct Teacher {
    stsrl::teacher::Leaf leaf{"value_net"};
    bool random_move = true;
    bool oracle = false;
};

Teacher parse_teacher(const Json& input) {
    Teacher teacher;
    if (!input.contains("teacher")) return teacher;
    for (const auto& [key, value] : input.at("teacher").items()) {
        if (key == "leaf") teacher.leaf.kind = value.get<std::string>();
        else if (key == "rollout_turns") teacher.leaf.rollout_turns = value.get<int>();
        else if (key == "rollout_steps") teacher.leaf.rollout_steps = value.get<int>();
        else if (key == "random_move") teacher.random_move = value.get<bool>();
        else if (key == "oracle") teacher.oracle = value.get<bool>();
        else throw std::invalid_argument{"unknown teacher setting: " + key};
    }
    return teacher;
}

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto teacher = parse_teacher(input);
    const auto search = stsrl::teacher::leaf_search(teacher.leaf, net);  // validates leaf vs net
    std::vector<Json> rows;
    const auto fight = stsrl::replay::to_fight(input, [&](const sts::BattleContext& start, const Json& columns) {
        return stsrl::teacher::play_fight(start, columns, rows, search, teacher.random_move, teacher.oracle);
    });
    const auto episode = fight.at("episode_id");
    auto settings = stsrl::teacher::settings(teacher.leaf, teacher.oracle);
    settings["random_move"] = teacher.random_move;
    return {{"episode_id", episode}, {"teacher", settings}, {"rows", rows}};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "value_play_worker", play); }
