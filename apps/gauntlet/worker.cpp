// Gauntlet card-reward evaluation (experiments/act-1-card-selection/approaches/gauntlet). The reward state is an
// act 1 run replayed through its first `actions.size()` fights (stored chosen actions; SimpleAgent out of combat,
// apps/common/fight_replay.hpp), then advanced to its card choice (the last card reward group; the potion / relic /
// key rewards SimpleAgent takes first are taken).
//   gauntlet_worker REQUEST.json OUTPUT_DIR [WEIGHTS]     (apps/common/worker.hpp)
//   REQUEST.json: {mode, run_seed, ascension, actions: [[chosen_action...] per replayed fight], ...}
//   mode "describe" -> {card_reward, offer, baseline, floor, hp, max_hp, boss, deck, elite_distances, boss_picks}
//     offer: the offered cards; options are 0..offer-1 and -1 = skip. baseline: SimpleAgent's option.
//     elite_distances: for each reachable elite node, the fewest combats (monster / elite rooms) before it,
//     distinct and ascending; boss_picks: the fewest combats before the boss. Each combat = one card pick.
//   mode "fight" + {encounter, picks, seed, teacher (apps/common/teacher_request.hpp)}
//     -> {teacher, results: [{option, picked, deck_size, won, start_hp, final_hp, pre, seconds}] per option, skip last}
//     pre: the persistent state right before BattleContext::init (apps/common/game_state.hpp).
//     Per option: take it, make `picks` SimpleAgent picks from fresh monster card rewards (card RNG = seed),
//     then play `encounter` (elite or boss room; every combat RNG from seed) from the reward state's HP, relics and
//     potions. Every option shares the seed, so the offers and the fight's RNG streams start identical.
#include "agents/teacher_search.hpp"
#include "apps/common/fight_replay.hpp"
#include "apps/common/game_state.hpp"
#include "apps/common/teacher_request.hpp"
#include "apps/common/worker.hpp"
#include "constants/MonsterEncounters.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

std::string card_name(const sts::Card& card) {
    std::string name = sts::cardEnumStrings[static_cast<int>(card.id)];
    for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return card.isUpgraded() ? name + "+" : name;
}

std::string encounter_name(sts::MonsterEncounter encounter) {
    std::string name = sts::monsterEncounterEnumNames[static_cast<int>(encounter)];
    for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return name;
}

constexpr std::array act1_targets{sts::MonsterEncounter::GREMLIN_NOB, sts::MonsterEncounter::LAGAVULIN,
                                  sts::MonsterEncounter::THREE_SENTRIES, sts::MonsterEncounter::SLIME_BOSS,
                                  sts::MonsterEncounter::THE_GUARDIAN, sts::MonsterEncounter::HEXAGHOST};

sts::MonsterEncounter encounter_from_name(const std::string& name) {
    for (const auto encounter : act1_targets)
        if (encounter_name(encounter) == name) return encounter;
    throw std::invalid_argument{"not an act 1 elite or boss: " + name};
}

// SimpleAgent's next rewards action is the card choice (stepRewardsScreen takes potions, relics and keys first).
bool card_choice_next(const sts::GameContext& game) {
    const auto& r = game.info.rewardsContainer;
    return game.screenState == sts::ScreenState::REWARDS && r.cardRewardCount > 0 &&
           !(r.potionCount > 0 && game.potionCount < game.potionCapacity) && r.relicCount == 0 && !r.sapphireKey &&
           !r.emeraldKey;
}

// The replayed run at its card choice; false if the run died, ended or has no card reward there.
bool reward_state(const Json& input, sts::GameContext& game) {
    const auto actions = input.at("actions").get<std::vector<std::vector<std::size_t>>>();
    if (actions.empty()) throw std::invalid_argument{"actions: replay at least one fight"};
    game = sts::GameContext{sts::CharacterClass::IRONCLAD, input.at("run_seed").get<std::uint64_t>(),
                            input.at("ascension").get<int>()};
    const auto fights = static_cast<int>(actions.size());
    const auto run = stsrl::act1::play(game, [&](const sts::BattleContext& start, const Json& fight) {
        return stsrl::replay::stored_fight(start, actions.at(fight.at("fight_index").get<std::size_t>()));
    }, fights);
    if (run.fights != fights) throw std::runtime_error{"run ended before its last replayed fight"};
    if (game.outcome != sts::GameOutcome::UNDECIDED || game.act != 1) return false;
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    while (game.screenState == sts::ScreenState::REWARDS && !card_choice_next(game)) agent.stepOutOfCombat(game);
    return card_choice_next(game);
}

const sts::CardReward& offer_of(const sts::GameContext& game) {
    const auto& r = game.info.rewardsContainer;
    return r.cardRewards[r.cardRewardCount - 1];
}

// The card the deck gained (by id), or "skip".
std::string added_card(const sts::Deck& before, const sts::Deck& after) {
    std::map<sts::CardId, int> count;
    for (const auto& card : before.cards) --count[card.id];
    for (const auto& card : after.cards)
        if (++count[card.id] > 0) return card_name(card);
    return "skip";
}

// SimpleAgent's pick from `offer`, applied to game: the added card or "skip".
std::string simple_pick(sts::GameContext& game, const sts::CardReward& offer) {
    game.info.rewardsContainer = sts::Rewards{offer};
    game.screenState = sts::ScreenState::REWARDS;
    game.regainControlAction = [](sts::GameContext&) {};  // skipping must not leave the (copied) reward screen
    const auto before = game.deck;
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    agent.stepCardReward(game);
    return added_card(before, game.deck);
}

bool is_combat(sts::Room room) { return room == sts::Room::MONSTER || room == sts::Room::ELITE; }

// Fewest combats before each reachable node from the current one (INT_MAX: unreachable), rows above it only.
std::array<std::array<int, 7>, 15> combats_before(const sts::GameContext& game) {
    std::array<std::array<int, 7>, 15> best;
    for (auto& row : best) row.fill(INT_MAX);
    const auto& map = *game.map;
    const auto& here = map.getNode(game.curMapNodeX, game.curMapNodeY);
    for (int e = 0; e < here.edgeCount; ++e) best[game.curMapNodeY + 1][here.edges[e]] = 0;
    for (int y = game.curMapNodeY + 1; y < 14; ++y)
        for (int x = 0; x < 7; ++x) {
            if (best[y][x] == INT_MAX) continue;
            const auto& node = map.getNode(x, y);
            const int through = best[y][x] + is_combat(node.room);
            for (int e = 0; e < node.edgeCount; ++e) best[y + 1][node.edges[e]] = std::min(best[y + 1][node.edges[e]], through);
        }
    return best;
}

Json describe(const Json& input) {
    sts::GameContext game;
    if (!reward_state(input, game)) return {{"card_reward", false}};
    if (game.curMapNodeY < 0 || game.curMapNodeY >= 14) throw std::runtime_error{"card reward off the act 1 map rows"};
    const auto best = combats_before(game);
    std::vector<int> elites;
    int boss = INT_MAX;
    for (int x = 0; x < 7; ++x) {
        for (int y = game.curMapNodeY + 1; y < 15; ++y)
            if (best[y][x] != INT_MAX && game.map->getNode(x, y).room == sts::Room::ELITE) elites.push_back(best[y][x]);
        if (best[14][x] != INT_MAX) boss = std::min(boss, best[14][x] + is_combat(game.map->getNode(x, 14).room));
    }
    std::sort(elites.begin(), elites.end());
    elites.erase(std::unique(elites.begin(), elites.end()), elites.end());
    if (boss == INT_MAX) throw std::runtime_error{"no path to the boss"};
    Json offer = Json::array(), deck = Json::array();
    for (const auto& card : offer_of(game)) offer.push_back(card_name(card));
    for (const auto& card : game.deck.cards) deck.push_back(card_name(card));
    auto copy = game;
    const auto choice = simple_pick(copy, offer_of(game));
    int baseline = -1;
    for (std::size_t i = 0; i < offer.size(); ++i)
        if (offer[i] == choice) { baseline = static_cast<int>(i); break; }
    return {{"card_reward", true}, {"offer", offer}, {"baseline", baseline}, {"floor", game.floorNum},
            {"hp", game.curHp}, {"max_hp", game.maxHp}, {"boss", encounter_name(game.boss)}, {"deck", deck},
            {"elite_distances", elites}, {"boss_picks", boss}};
}

Json fight(const Json& input, const stsrl::ValueNet* net) {
    const auto teacher = stsrl::teacher::parse_request(input.at("teacher"));
    if (teacher.oracle) throw std::invalid_argument{"the gauntlet plays fair: oracle must be false"};
    const auto search = stsrl::teacher::leaf_search(teacher.leaf, net, teacher.budget);
    const auto encounter = encounter_from_name(input.at("encounter").get<std::string>());
    const bool boss = encounter == sts::MonsterEncounter::SLIME_BOSS || encounter == sts::MonsterEncounter::THE_GUARDIAN ||
                      encounter == sts::MonsterEncounter::HEXAGHOST;
    const auto picks = input.at("picks").get<int>();
    const auto seed = input.at("seed").get<std::uint64_t>();
    if (picks < 0) throw std::invalid_argument{"picks must be >= 0"};
    sts::GameContext reward;
    if (!reward_state(input, reward)) throw std::runtime_error{"no card reward at this state"};
    const auto offer = offer_of(reward);
    Json results = Json::array();
    for (int option = 0; option <= static_cast<int>(offer.size()); ++option) {
        const int chosen = option == static_cast<int>(offer.size()) ? -1 : option;
        const auto began = std::chrono::steady_clock::now();
        auto game = reward;
        game.info.rewardsContainer.clear();
        if (chosen >= 0) game.obtainCard(offer[chosen]);
        game.cardRng = sts::Random{seed};
        Json picked = Json::array();
        for (int i = 0; i < picks; ++i) picked.push_back(simple_pick(game, game.createCardReward(sts::Room::MONSTER)));
        game.curRoom = boss ? sts::Room::BOSS : sts::Room::ELITE;
        game.seed = seed;  // BattleContext::init seeds aiRng, monsterHpRng, shuffleRng, cardRandomRng from seed + floor
        game.miscRng = sts::Random{seed};
        game.potionRng = sts::Random{seed};
        const auto pre = stsrl::game_state::state(game);  // pre-init state: the outcome model's input
        sts::BattleContext battle;
        battle.init(game, encounter);
        const int start_hp = battle.player.curHp;
        std::vector<Json> rows;  // not kept
        const auto end = stsrl::teacher::play_fight(battle, {{"episode_id", seed}}, rows, search, false, false,
                                                    teacher.budget.particles, false);
        const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
        results.push_back({{"option", chosen}, {"picked", picked}, {"deck_size", game.deck.size()}, {"won", won}, {"pre", pre},
                           {"start_hp", start_hp}, {"final_hp", won ? static_cast<int>(end.player.curHp) : 0},
                           {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count()}});
    }
    auto settings = stsrl::teacher::settings(teacher.leaf, teacher.oracle, teacher.budget);
    return {{"teacher", settings}, {"results", results}};
}

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto mode = input.at("mode").get<std::string>();
    if (mode == "describe") return describe(input);
    if (mode == "fight") return fight(input, net);
    throw std::invalid_argument{"mode must be describe or fight"};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "gauntlet_worker", play); }
