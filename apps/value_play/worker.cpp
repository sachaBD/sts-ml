// One fight of a combat_v3 data run, replayed with a teacher leaf strategy. The act 1 run is rebuilt
// up to that fight: SimpleAgent out of combat (as in bootstrap), the earlier fights by their stored
// chosen actions. Then the teacher plays the fight and the worker stops.
//   value_play_worker WEIGHTS FIGHT.json OUTPUT_DIR
//   value_play_worker --no-weights FIGHT.json OUTPUT_DIR     (guided_rollout leaf: no value net)
//   FIGHT.json: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight],
//                teacher (optional): {leaf, rollout_turns, rollout_steps, random_move}}
//   teacher defaults: leaf value_net, rollout bounds 0, random_move true (the original value_play).
//   Invalid combinations (e.g. hybrid without bounds, guided_rollout with weights) fail.
// Output: OUTPUT_DIR/fight.msgpack = {episode_id, teacher, rows} (combat_v3 rows of that fight);
// teacher = teacher::settings(leaf) + random_move.
#include "agents/teacher_search.hpp"
#include "apps/fight_replay.hpp"
#include "apps/worker_io.hpp"

#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;

struct Teacher {
    stsrl::teacher::Leaf leaf{"value_net"};
    bool random_move = true;
};

Teacher parse_teacher(const Json& input) {
    Teacher teacher;
    if (!input.contains("teacher")) return teacher;
    for (const auto& [key, value] : input.at("teacher").items()) {
        if (key == "leaf") teacher.leaf.kind = value.get<std::string>();
        else if (key == "rollout_turns") teacher.leaf.rollout_turns = value.get<int>();
        else if (key == "rollout_steps") teacher.leaf.rollout_steps = value.get<int>();
        else if (key == "random_move") teacher.random_move = value.get<bool>();
        else throw std::invalid_argument{"unknown teacher setting: " + key};
    }
    return teacher;
}

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto teacher = parse_teacher(input);
    const auto search = stsrl::teacher::leaf_search(teacher.leaf, net);  // validates leaf vs net
    std::vector<Json> rows;
    const auto fight = stsrl::replay::to_fight(input, [&](const sts::BattleContext& start, const Json& columns) {
        return stsrl::teacher::play_fight(start, columns, rows, search, teacher.random_move);
    });
    const auto episode = fight.at("episode_id");
    auto settings = stsrl::teacher::settings(teacher.leaf);
    settings["random_move"] = teacher.random_move;
    return {{"episode_id", episode}, {"teacher", settings}, {"rows", rows}};
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: value_play_worker WEIGHTS|--no-weights FIGHT.json OUTPUT_DIR\n";
        return 2;
    }
    try {
        std::optional<stsrl::ValueNet> net;
        if (std::string_view{argv[1]} != "--no-weights") net.emplace(argv[1]);
        std::ifstream file{argv[2]};
        if (!file) throw std::runtime_error{std::string{"cannot read "} + argv[2]};
        stsrl::worker::write_result(argv[3], play(Json::parse(file), net ? &*net : nullptr));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "value_play worker (" << argv[2] << "): " << error.what() << '\n';
        return 1;
    }
}
