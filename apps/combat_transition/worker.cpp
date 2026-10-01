// Combat transitions of a stored combat_v3 act 1 run (apps/combat_transition/extract.py): the run is replayed from
// its seed (SimpleAgent out of combat, each fight by its stored chosen actions, apps/common/fight_replay.hpp) and
// every fight's persistent game state is recorded right before BattleContext::init and right after exitBattle
// (end-of-combat relics such as Burning Blood applied; before the reward screen).
//   combat_transition_worker REQUEST.json OUTPUT_DIR     (apps/common/worker.hpp)
//   REQUEST.json: {run_seed, ascension, fights: [{actions: [chosen_action...], won, final_hp, potions, encounter}...]}
//   result: {fights: [{fight_index, replay: "ok" | "diverged", reason, start, pre: STATE, post: STATE, ...}]}
//   start: the replayed decision 0 encoding (global_numeric, cards, monsters) for check_start.
// The replay stops at the first fight that diverges from its stored outcome (simulator drift): that fight is
// returned with replay = "diverged" and no states; later fights are not returned.
#include "apps/common/fight_replay.hpp"
#include "apps/common/worker.hpp"
#include "combat/encoding.hpp"
#include "apps/common/game_state.hpp"

#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using stsrl::game_state::state;

Json play(const Json& input, const stsrl::ValueNet*) {
    const auto& stored = input.at("fights");
    const auto count = static_cast<int>(stored.size());
    sts::GameContext game{sts::CharacterClass::IRONCLAD, input.at("run_seed").get<std::uint64_t>(),
                          input.at("ascension").get<int>()};
    Json fights = Json::array();
    struct Diverged {};
    try {
        stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json& columns) {
            const auto index = columns.at("fight_index").get<int>();
            const auto& want = stored.at(static_cast<std::size_t>(index));
            Json fight = columns;
            const auto fail = [&](const std::string& reason) {
                fight["replay"] = "diverged";
                fight["reason"] = reason;
                fights.push_back(fight);
                throw Diverged{};
            };
            if (columns.at("encounter") != want.at("encounter")) fail("encounter differs");
            const auto pre = state(game);
            {  // the replayed decision 0 encoding, for check_start against the stored row (apps/common/replay.py)
                const auto row = stsrl::state_row(stsrl::CombatEnvironment{start}.decision().encoding, false);
                fight["start"] = {{"global_numeric", row.at("global_numeric")}, {"cards", row.at("cards")},
                                  {"monsters", row.at("monsters")}};
            }
            sts::BattleContext end;
            try {
                end = stsrl::replay::stored_fight(start, want.at("actions").get<std::vector<std::size_t>>());
            } catch (const std::runtime_error& e) {
                fail(e.what());
            }
            const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
            const int final_hp = won ? static_cast<int>(end.player.curHp) : 0;
            if (won != want.at("won").get<bool>() || final_hp != want.at("final_hp").get<int>()) fail("outcome differs");
            if (end.potionCount != want.at("potions").get<int>()) fail("potion count differs");
            auto after = game;
            end.exitBattle(after);
            fight["replay"] = "ok";
            fight["won"] = won;
            fight["battle_final_hp"] = static_cast<int>(end.player.curHp);  // = combat_v3 final_hp on a win
            fight["escaped"] = end.escapedCombat;
            fight["battle_potions"] = end.potionCount;
            fight["pre"] = pre;
            fight["post"] = state(after);
            fights.push_back(fight);
            return end;
        }, count);
    } catch (const Diverged&) {
    }
    return {{"fights", fights}};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "combat_transition_worker", play); }
