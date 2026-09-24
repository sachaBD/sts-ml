// One DAgger fight (slop_docs/apps/dagger.md): a stored combat_v3 bootstrap fight is rebuilt
// (apps/common/fight_replay.hpp), then the frozen value-net learner plays it while the guided-rollout teacher
// labels every reached decision (teacher::play_learner_fight).
//   dagger_worker REQUEST.json OUTPUT_DIR WEIGHTS     (apps/common/worker.hpp; WEIGHTS: the learner)
//   REQUEST.json: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight], max_decisions,
//                max_turns}
// Output: OUTPUT_DIR/result.msgpack = {episode_id, status, decisions, start, diagnostics, learner, teacher, rows}
//   status completed: rows = the fight's combat_v3 decision rows; capped (max_decisions) / turn_limit
//   (max_turns reached): no rows.
//   start: decision 0's row without outcome columns; diagnostics: per-decision learner/teacher moves and counters.
#include "agents/teacher_search.hpp"
#include "apps/common/fight_replay.hpp"
#include "apps/common/worker.hpp"

#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::json;
namespace teacher = stsrl::teacher;

Json play(const Json& input, const stsrl::ValueNet* net) {
    if (!net) throw std::invalid_argument{"dagger_worker needs WEIGHTS (the learner's value net)"};
    const teacher::Leaf learner_leaf{"value_net"}, teacher_leaf{"guided_rollout"};
    const auto learner = teacher::leaf_search(learner_leaf, net);
    const auto labels = teacher::leaf_search(teacher_leaf, nullptr);
    const auto max_decisions = input.at("max_decisions").get<int>();
    const auto max_turns = input.at("max_turns").get<int>();
    std::vector<Json> rows;
    teacher::LearnerFight result;
    Json fight;
    // A capped fight stops the replay here: its unfinished battle must not be exited into the act 1 run.
    struct Capped {};
    try {
        stsrl::replay::to_fight(input, [&](const sts::BattleContext& start, const Json& columns) {
            fight = columns;
            result = teacher::play_learner_fight(start, columns, rows, learner, labels, max_decisions, max_turns);
            if (!result.completed) throw Capped{};
            return result.battle;
        });
    } catch (const Capped&) {
    }
    auto learner_settings = teacher::search_settings(learner_leaf);
    learner_settings["random_move"] = false;
    return {{"episode_id", fight.at("episode_id")}, {"status", result.status},
            {"decisions", result.decisions.size()}, {"start", result.start}, {"diagnostics", result.decisions},
            {"learner", learner_settings}, {"teacher", teacher::search_settings(teacher_leaf)}, {"rows", rows}};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "dagger_worker", play); }
