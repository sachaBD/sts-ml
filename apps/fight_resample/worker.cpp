// Resamples one stored fight of a combat_v3 bootstrap run: the act 1 run is replayed up to that fight
// (apps/common/fight_replay.hpp), then the teacher plays it once per sample from the same deck, relics and
// potions, but with the sample's starting HP and a fresh fight: every combat RNG is seeded from the
// sample's episode_id (draw order, monster HP and AI, ...).
//   fight_resample_worker REQUEST.json OUTPUT_DIR [WEIGHTS]     (apps/common/worker.hpp)
//   WEIGHTS: the value net, for the value_net / hybrid leaves.
//   REQUEST.json: {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight],
//                random_potions, samples: [{episode_id, starting_hp}], teacher: apps/common/teacher_request.hpp}
//   random_potions: each potion the player holds is replaced by a random potion drop (potion RNG seeded as above).
// Output: OUTPUT_DIR/result.msgpack = {teacher, rows} (combat_v3 rows of every sample, in sample order).
#include "agents/teacher_search.hpp"
#include "apps/common/fight_replay.hpp"
#include "apps/common/teacher_request.hpp"
#include "apps/common/worker.hpp"
#include "game/Game.h"

#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::json;

// The game right before the fight, with the sample's HP and every combat RNG seeded from its episode_id.
sts::GameContext resampled(sts::GameContext game, std::uint64_t episode_id, int hp, bool random_potions) {
    if (hp < 1 || hp > game.maxHp) throw std::invalid_argument{"starting_hp must be in [1, max_hp]"};
    game.curHp = hp;
    game.seed = episode_id;  // BattleContext::init seeds aiRng, monsterHpRng, shuffleRng, cardRandomRng from seed + floor
    game.miscRng = sts::Random{episode_id};
    game.potionRng = sts::Random{episode_id};
    if (random_potions)
        for (int i = 0; i < game.potionCapacity; ++i)
            if (game.potions[i] != sts::Potion::EMPTY_POTION_SLOT)
                game.potions[i] = sts::returnRandomPotion(game.potionRng, game.cc);
    return game;
}

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto teacher = stsrl::teacher::parse_request(input.at("teacher"));
    const auto search = stsrl::teacher::leaf_search(teacher.leaf, net, teacher.budget);  // validates leaf vs net
    const auto random_potions = input.at("random_potions").get<bool>();
    std::vector<Json> rows;
    stsrl::replay::to_fight(input, [&](const sts::GameContext& game, const sts::BattleContext& start, const Json& source) {
        for (const auto& sample : input.at("samples")) {
            const auto episode_id = sample.at("episode_id").get<std::uint64_t>();
            sts::BattleContext battle;
            battle.init(resampled(game, episode_id, sample.at("starting_hp").get<int>(), random_potions));
            auto fight = source;
            fight.update({{"episode_id", episode_id}, {"source_episode_id", source.at("episode_id")},
                          {"starting_hp", battle.player.curHp}});
            stsrl::teacher::play_fight(battle, fight, rows, search, teacher.random_move, teacher.oracle,
                                        teacher.budget.particles);
        }
        return start;  // the replayed run stops after this fight
    });
    auto settings = stsrl::teacher::settings(teacher.leaf, teacher.oracle, teacher.budget);
    settings["random_move"] = teacher.random_move;
    return {{"teacher", settings}, {"rows", rows}};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "fight_resample_worker", play); }
