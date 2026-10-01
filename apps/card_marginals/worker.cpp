// card_marginals: synthetic pre-combat states with paired card additions (apps/card_marginals/card_marginals.py).
// A deck group is built from its group_seed by the simulator itself; only counts are sampled:
//   both stages: a fresh A20 Ironclad run (seed = group_seed); SimpleAgent plays Neow (option 0, as in the natural
//   cohort). Card reward picks: SimpleAgent with simple_pick_prob, else uniformly random among the offer or skip
//   with skip_prob.
//   stage easy: prior_fights ~ prior_fights_weights; per prior fight a potion reward roll (addPotionRewards, kept
//     if a slot is free) and a monster card reward; HP drawn from hp_by_prior[prior_fights] (natural HP).
//   stage hard_elite / boss: the counts of `donor` (a natural hard / elite, or boss, pre-fight state, chosen by the
//     driver): prior fights (elites_before of them elite rooms at random positions: elite card reward roll + elite-tier relic),
//     extra relics (act 1 tier roll 50/33/17), removed Strikes / Defends and upgraded cards (each up to the donor's
//     count, counting Neow's; SimpleAgent's card-select choice with simple_pick_prob, else a random Strike / Defend /
//     upgradable card), potions (random, up to the donor's count), HP fraction and floor. Relics are obtained with
//     their pickup effects; a pickup screen (bottles) is played by SimpleAgent. Persistent combat relic counters
//     (generator relic_counters: relic id -> natural values, e.g. Pen Nib, Nunchaku) get a natural value.
//   Variants: the base deck ("skip") and the base deck + each of n_candidates distinct candidate cards: one card
//   of a fresh monster card reward roll from the deck's state (natural rarity rates), or, with uniform_frac, one
//   uniform over the Ironclad class pool (common / uncommon / rare).
// Every variant plays the same fights: per encounter, `seeds` fight seeds (the fight's RNG streams).
//   card_marginals_worker REQUEST.json OUTPUT_DIR     (apps/common/worker.hpp)
//   REQUEST.json: {mode, stage, ascension, group_seed, generator: {...}, [donor],
//                  [encounters, seeds, teachers: {easy|hard|elite|boss: teacher request}]}
//   A fight's room (relic triggers such as Preserved Insect, Pantograph): ELITE / BOSS / MONSTER by its kind.
//   mode "deck"  -> {group}
//   mode "fight" -> {group, teachers, results: [{encounter, kind, seed_index, seed, variant, card, source, pre, won,
//                    start_hp, battle_final_hp, post, simulations, seconds}]}
//   group: {group_seed, prior_fights, neow, rewards: [{room, offer, pick, by, relic}], potion_rolls, relics_added,
//           removes, upgrades, hp, max_hp, floor, deck, relics, potions, candidates: [{card, source}]}
//   pre: the persistent state right before BattleContext::init (apps/common/game_state.hpp); post: after exitBattle
//   (end-of-combat relics such as Burning Blood applied). battle_final_hp: in-battle HP before exitBattle (0 if lost).
#include "agents/teacher_search.hpp"
#include "apps/common/game_state.hpp"
#include "apps/common/teacher_request.hpp"
#include "apps/common/worker.hpp"
#include "constants/CardPools.h"
#include "constants/MonsterEncounters.h"
#include "constants/Relics.h"
#include "game/Game.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string card_name(const sts::Card& card) {
    const auto name = lower(sts::cardEnumStrings[static_cast<int>(card.id)]);
    return card.isUpgraded() ? name + "+" : name;
}

Json deck_names(const sts::Deck& deck) {
    Json names = Json::array();
    for (const auto& card : deck.cards) names.push_back(card_name(card));
    return names;
}

std::string encounter_name(sts::MonsterEncounter encounter) {
    return lower(sts::monsterEncounterEnumNames[static_cast<int>(encounter)]);
}

// An act 1 easy-pool, hard-pool, elite or boss encounter by name, and its kind (easy / hard / elite / boss).
std::pair<sts::MonsterEncounter, std::string> act1_encounter(const std::string& name) {
    namespace pool = sts::MonsterEncounterPool;
    for (int i = 0; i < pool::weakCount[0]; ++i)
        if (encounter_name(pool::weakEnemies[0][i]) == name) return {pool::weakEnemies[0][i], "easy"};
    for (int i = 0; i < pool::strongCount[0]; ++i)
        if (encounter_name(pool::strongEnemies[0][i]) == name) return {pool::strongEnemies[0][i], "hard"};
    for (const auto e : {sts::MonsterEncounter::GREMLIN_NOB, sts::MonsterEncounter::LAGAVULIN, sts::MonsterEncounter::THREE_SENTRIES})
        if (encounter_name(e) == name) return {e, "elite"};
    for (const auto e : {sts::MonsterEncounter::SLIME_BOSS, sts::MonsterEncounter::THE_GUARDIAN, sts::MonsterEncounter::HEXAGHOST})
        if (encounter_name(e) == name) return {e, "boss"};
    throw std::invalid_argument{"not an act 1 easy, hard, elite or boss encounter: " + name};
}

std::uint64_t mix(std::uint64_t x) {  // splitmix64 finaliser
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// The card the deck gained (by id, first surplus copy), or "skip".
std::string added_card(const sts::Deck& before, const sts::Deck& after) {
    std::map<sts::CardId, int> count;
    for (const auto& card : before.cards) --count[card.id];
    for (const auto& card : after.cards)
        if (++count[card.id] > 0) return card_name(card);
    return "skip";
}

// SimpleAgent's pick from `offer` (as apps/gauntlet/worker.cpp), applied to game: the added card or "skip".
std::string simple_pick(sts::GameContext& game, const sts::CardReward& offer) {
    const auto screen = game.screenState;
    game.info.rewardsContainer = sts::Rewards{};
    game.info.rewardsContainer.addCardReward(offer);
    game.screenState = sts::ScreenState::REWARDS;
    game.regainControlAction = [](sts::GameContext&) {};  // skipping must not leave the (synthetic) reward screen
    const auto before = game.deck;
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    agent.stepCardReward(game);
    game.info.rewardsContainer.clear();
    game.screenState = screen;
    return added_card(before, game.deck);
}

double uniform01(std::mt19937_64& rng) { return std::uniform_real_distribution<double>{0.0, 1.0}(rng); }

template <class T>
const T& pick(std::mt19937_64& rng, const std::vector<T>& values) {
    return values.at(std::uniform_int_distribution<std::size_t>{0, values.size() - 1}(rng));
}

struct Candidate {
    sts::Card card;
    std::string source;  // reward | uniform
};

struct Group {
    sts::GameContext game;  // the base state (map screen, no reward pending)
    std::vector<Candidate> candidates;
    Json description;
};

double number(const Json& g, const char* key, double lo, double hi) {
    const auto v = g.at(key).get<double>();
    if (!(v >= lo && v <= hi)) throw std::invalid_argument{std::string{key} + " out of range"};
    return v;
}

Json potion_names(const sts::GameContext& game) {
    Json potions = Json::array();
    for (int i = 0; i < game.potionCapacity; ++i)
        if (game.potions[i] != sts::Potion::EMPTY_POTION_SLOT && game.potions[i] != sts::Potion::INVALID)
            potions.push_back(lower(sts::potionEnumNames[static_cast<int>(game.potions[i])]));
    return potions;
}

Json relic_names(const sts::GameContext& game) {
    Json relics = Json::array();
    for (const auto& r : game.relics.relics) relics.push_back(lower(sts::relicEnumNames[static_cast<int>(r.id)]));
    return relics;
}

int basics(const sts::Deck& deck) {
    return static_cast<int>(std::count_if(deck.cards.begin(), deck.cards.end(), [](const sts::Card& c) {
        return c.id == sts::CardId::STRIKE_RED || c.id == sts::CardId::DEFEND_RED;
    }));
}


int upgraded(const sts::Deck& deck) {
    return static_cast<int>(std::count_if(deck.cards.begin(), deck.cards.end(), [](const sts::Card& c) { return c.isUpgraded(); }));
}

// Plays any screen the game opened (e.g. a bottle relic's card select) with SimpleAgent, back to the map screen.
void settle(sts::GameContext& game) {
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    for (int steps = 0; game.screenState != sts::ScreenState::MAP_SCREEN; ++steps) {
        if (steps > 50) throw std::runtime_error{"a pickup screen did not return to the map"};
        agent.stepOutOfCombat(game);
    }
}

void back_to_map(sts::GameContext& game) {
    game.screenState = sts::ScreenState::MAP_SCREEN;
    game.regainControlAction = [](sts::GameContext& g) { g.screenState = sts::ScreenState::MAP_SCREEN; };
}

std::string obtain_relic(sts::GameContext& game, sts::RelicId relic) {
    back_to_map(game);
    game.obtainRelic(relic);
    settle(game);
    return lower(sts::relicEnumNames[static_cast<int>(relic)]);
}

bool is_basic(const sts::Card& c) { return c.id == sts::CardId::STRIKE_RED || c.id == sts::CardId::DEFEND_RED; }

// SimpleAgent's choice on a one-card select screen, as at a shop or rest site: upgrade (any upgradable card) or
// remove (among Strikes / Defends only: the removes a donor's count stands for).
void simple_select(sts::GameContext& game, sts::CardSelectScreenType type) {
    back_to_map(game);
    game.openCardSelectScreen(type, 1);
    if (type == sts::CardSelectScreenType::REMOVE) {
        game.info.toSelectCards.clear();
        for (int i = 0; i < game.deck.size(); ++i)
            if (is_basic(game.deck.cards[i])) game.info.toSelectCards.push_back({game.deck.cards[i], i});
    }
    if (game.info.toSelectCards.size() == 0) {
        game.screenState = sts::ScreenState::MAP_SCREEN;
        return;
    }
    settle(game);
}

// A card reward for one prior fight, picked (SimpleAgent / random / skip) and applied.
Json reward(sts::GameContext& game, sts::Room room, std::mt19937_64& rng, double simple_prob, double skip_prob) {
    const auto offer = game.createCardReward(room);
    Json names = Json::array();
    for (const auto& card : offer) names.push_back(card_name(card));
    std::string chosen, by;
    if (uniform01(rng) < simple_prob) {
        chosen = simple_pick(game, offer), by = "simple";
    } else if (uniform01(rng) < skip_prob || offer.size() == 0) {
        chosen = "skip", by = "random";
    } else {
        const auto i = std::uniform_int_distribution<int>{0, static_cast<int>(offer.size()) - 1}(rng);
        game.obtainCard(offer[i]);
        chosen = card_name(offer[i]), by = "random";
    }
    return {{"room", room == sts::Room::ELITE ? "elite" : "monster"}, {"offer", names}, {"pick", chosen}, {"by", by}};
}

Group build_group(const Json& input) {
    const auto stage = input.at("stage").get<std::string>();
    if (stage != "easy" && stage != "hard_elite" && stage != "boss")
        throw std::invalid_argument{"stage must be easy, hard_elite or boss"};
    const auto& g = input.at("generator");
    const auto group_seed = input.at("group_seed").get<std::uint64_t>();
    const double simple_prob = number(g, "simple_pick_prob", 0, 1), skip_prob = number(g, "skip_prob", 0, 1),
                 uniform_frac = number(g, "uniform_frac", 0, 1);
    const auto n_candidates = g.at("n_candidates").get<int>();
    if (n_candidates < 0 || n_candidates > 60) throw std::invalid_argument{"need 0 <= n_candidates <= 60"};
    std::mt19937_64 rng{mix(group_seed ^ 0xc4d5a11ULL)};  // generator choices (the game has its own RNG streams)

    Group group;
    auto& game = group.game;
    game = sts::GameContext{sts::CharacterClass::IRONCLAD, group_seed, input.at("ascension").get<int>()};
    const auto neow = game.info.neowRewards[0];
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    for (int steps = 0; game.screenState != sts::ScreenState::MAP_SCREEN; ++steps) {
        if (steps > 100 || game.outcome != sts::GameOutcome::UNDECIDED) throw std::runtime_error{"Neow did not reach the map"};
        agent.stepOutOfCombat(game);
    }
    Json neow_json = {{"bonus", sts::Neow::bonusStrings[static_cast<int>(neow.r)]},
                      {"drawback", sts::Neow::drawbackStrings[static_cast<int>(neow.d)]}, {"deck_after", deck_names(game.deck)}};

    int prior = 0;
    Json rewards = Json::array(), potion_rolls = Json::array(), relics_added = Json::array(), extra = Json::object();
    if (stage == "easy") {
        const auto weights = g.at("prior_fights_weights").get<std::vector<double>>();
        const auto hp_by_prior = g.at("hp_by_prior").get<std::vector<std::vector<int>>>();
        if (weights.empty() || weights.size() != hp_by_prior.size())
            throw std::invalid_argument{"need prior_fights_weights and hp_by_prior of equal length"};
        prior = static_cast<int>(std::discrete_distribution<int>{weights.begin(), weights.end()}(rng));
        for (int f = 0; f < prior; ++f) {
            sts::Rewards potion;
            game.addPotionRewards(potion);
            if (potion.potionCount > 0) {
                const bool kept = game.potionCount < game.potionCapacity;
                if (kept) game.obtainPotion(potion.potions[0]);
                potion_rolls.push_back({{"potion", lower(sts::potionEnumNames[static_cast<int>(potion.potions[0])])}, {"kept", kept}});
            } else {
                potion_rolls.push_back(nullptr);
            }
            rewards.push_back(reward(game, sts::Room::MONSTER, rng, simple_prob, skip_prob));
        }
        game.floorNum = prior + 1;
        game.curHp = std::clamp(pick(rng, hp_by_prior.at(prior)), 1, game.maxHp);
    } else {
        const auto& donor = input.at("donor");
        prior = donor.at("fight_index").get<int>();
        const int elites = std::min(donor.at("elites_before").get<int>(), prior);
        std::vector<int> order(static_cast<std::size_t>(prior));
        for (int i = 0; i < prior; ++i) order[i] = i < elites;  // 1 = an elite room
        std::shuffle(order.begin(), order.end(), rng);
        for (int f = 0; f < prior; ++f) {
            const auto room = order[f] ? sts::Room::ELITE : sts::Room::MONSTER;
            std::string relic;
            if (order[f]) {
                relic = obtain_relic(game, game.returnRandomRelic(sts::returnRandomRelicTierElite(game.relicRng)));
                relics_added.push_back({{"relic", relic}, {"from", "elite"}});
            }
            auto r = reward(game, room, rng, simple_prob, skip_prob);
            if (order[f]) r["relic"] = relic;
            rewards.push_back(r);
        }
        for (int i = 0; i < donor.at("extra_relics").get<int>(); ++i)
            relics_added.push_back({{"relic", obtain_relic(game, game.returnRandomRelic(sts::returnRandomRelicTier(game.relicRng, 1)))},
                                    {"from", "extra"}});
        Json removes = Json::array(), upgrades = Json::array();
        const int keep_basics = 9 - donor.at("removed").get<int>();
        for (int tries = 0; basics(game.deck) > keep_basics && tries < 12; ++tries) {
            const auto before = game.deck;
            if (uniform01(rng) < simple_prob) {
                simple_select(game, sts::CardSelectScreenType::REMOVE);
            } else {
                std::vector<int> idx;
                for (int i = 0; i < game.deck.size(); ++i)
                    if (is_basic(game.deck.cards[i])) idx.push_back(i);
                game.deck.remove(game, pick(rng, idx));
            }
            std::map<std::string, int> diff;
            for (const auto& c : before.cards) ++diff[card_name(c)];
            for (const auto& c : game.deck.cards) --diff[card_name(c)];
            for (const auto& [name, n] : diff)
                if (n > 0) removes.push_back(name);
        }
        const int want_upgraded = donor.at("upgrades").get<int>();
        for (int tries = 0; upgraded(game.deck) < want_upgraded && game.deck.getUpgradeableCount() > 0 && tries < 20; ++tries) {
            const auto before = game.deck;
            if (uniform01(rng) < simple_prob) {
                simple_select(game, sts::CardSelectScreenType::UPGRADE);
            } else {
                std::vector<int> idx;
                for (int i = 0; i < game.deck.size(); ++i)
                    if (game.deck.cards[i].canUpgrade()) idx.push_back(i);
                game.deck.upgrade(pick(rng, idx));
            }
            for (int i = 0; i < game.deck.size() && i < before.size(); ++i)
                if (game.deck.cards[i].isUpgraded() != before.cards[i].isUpgraded()) upgrades.push_back(card_name(game.deck.cards[i]));
        }
        const int want_potions = std::min(donor.at("potions").get<int>(), game.potionCapacity);
        while (game.potionCount < want_potions) game.obtainPotion(sts::returnRandomPotion(game.potionRng, sts::CharacterClass::IRONCLAD));
        Json counters = Json::object();  // persistent relic counters: a natural value of the same relic
        if (g.contains("relic_counters"))
            for (auto& r : game.relics.relics)
                if (const auto key = std::to_string(static_cast<int>(r.id)); g.at("relic_counters").contains(key)) {
                    r.data = pick(rng, g.at("relic_counters").at(key).get<std::vector<int>>());
                    counters[lower(sts::relicEnumNames[static_cast<int>(r.id)])] = r.data;
                }
        game.floorNum = donor.at("floor").get<int>();
        const double fraction = donor.at("hp").get<double>() / donor.at("max_hp").get<double>();
        game.curHp = std::clamp(static_cast<int>(std::lround(fraction * game.maxHp)), 1, game.maxHp);
        extra = {{"removes", removes}, {"upgrades", upgrades}, {"relic_counters", counters}, {"donor", donor}};
    }

    std::vector<sts::CardId> ironclad;  // the class pool: common, uncommon, rare
    for (const auto rarity : {sts::CardRarity::COMMON, sts::CardRarity::UNCOMMON, sts::CardRarity::RARE})
        for (int i = 0; i < sts::RarityCardPool::getPoolSize(sts::CharacterClass::IRONCLAD, rarity); ++i)
            ironclad.push_back(sts::RarityCardPool::getCardFromPool(sts::CharacterClass::IRONCLAD, rarity, i));
    Json candidates = Json::array();
    for (int tries = 0; static_cast<int>(group.candidates.size()) < n_candidates; ++tries) {
        if (tries > 50 * (n_candidates + 1)) throw std::runtime_error{"cannot draw distinct candidates"};
        Candidate c{sts::Card{sts::CardId::INVALID}, ""};
        if (uniform01(rng) < uniform_frac) {
            c = {game.previewObtainCard(sts::Card{pick(rng, ironclad)}), "uniform"};
        } else {
            auto copy = game;  // a fresh roll from the deck's own reward state (rarity factor), own RNG draw
            copy.cardRng = sts::Random{rng()};
            const auto offer = copy.createCardReward(sts::Room::MONSTER);
            c = {offer[std::uniform_int_distribution<int>{0, static_cast<int>(offer.size()) - 1}(rng)], "reward"};
        }
        if (std::any_of(group.candidates.begin(), group.candidates.end(), [&](const Candidate& o) { return o.card.id == c.card.id; }))
            continue;
        group.candidates.push_back(c);
        candidates.push_back({{"card", card_name(c.card)}, {"source", c.source}});
    }
    back_to_map(game);
    game.info.rewardsContainer.clear();
    group.description = {{"group_seed", group_seed}, {"stage", stage}, {"prior_fights", prior}, {"neow", neow_json},
                         {"rewards", rewards}, {"potion_rolls", potion_rolls}, {"relics_added", relics_added},
                         {"hp", game.curHp}, {"max_hp", game.maxHp}, {"floor", game.floorNum}, {"deck", deck_names(game.deck)},
                         {"relics", relic_names(game)}, {"potions", potion_names(game)}, {"candidates", candidates}};
    group.description.update(extra);
    return group;
}

Json fight(const Json& input, const stsrl::ValueNet* net) {
    std::map<std::string, std::pair<stsrl::teacher::Request, stsrl::teacher::SearchFn>> teachers;
    Json settings = Json::object();
    for (const auto& [kind, request] : input.at("teachers").items()) {
        const auto teacher = stsrl::teacher::parse_request(request);
        if (teacher.oracle || teacher.random_move) throw std::invalid_argument{"card_marginals plays fair: no oracle, no random move"};
        teachers.emplace(kind, std::pair{teacher, stsrl::teacher::leaf_search(teacher.leaf, net, teacher.budget)});
        settings[kind] = stsrl::teacher::settings(teacher.leaf, teacher.oracle, teacher.budget);
    }
    const auto seeds = input.at("seeds").get<int>();
    if (seeds < 1) throw std::invalid_argument{"seeds must be >= 1"};
    const auto group = build_group(input);
    const auto group_seed = input.at("group_seed").get<std::uint64_t>();
    Json results = Json::array();
    for (const auto& name : input.at("encounters").get<std::vector<std::string>>()) {
        const auto [encounter, kind] = act1_encounter(name);
        if (!teachers.contains(kind)) throw std::invalid_argument{"no teacher for " + kind + " fights"};
        const auto& [teacher, search] = teachers.at(kind);
        for (int s = 0; s < seeds; ++s) {
            const auto seed = mix(mix(group_seed) ^ mix(static_cast<std::uint64_t>(encounter) * 1000003ULL + s)) >> 8;
            for (int v = -1; v < static_cast<int>(group.candidates.size()); ++v) {
                const auto began = std::chrono::steady_clock::now();
                auto game = group.game;
                if (v >= 0) game.obtainCard(group.candidates[v].card);
                game.curRoom = kind == "boss" ? sts::Room::BOSS : kind == "elite" ? sts::Room::ELITE : sts::Room::MONSTER;
                game.seed = seed;  // BattleContext::init seeds aiRng, monsterHpRng, shuffleRng, cardRandomRng from seed + floor
                game.miscRng = sts::Random{seed};
                game.potionRng = sts::Random{seed};
                const auto pre = stsrl::game_state::state(game);
                sts::BattleContext battle;
                battle.init(game, encounter);
                const int start_hp = battle.player.curHp;
                // records nothing: only the outcome is kept (forced moves are played without a search)
                const auto end = stsrl::teacher::play_fight(battle, search, false, teacher.budget.particles, false);
                const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
                auto after = game;
                after.regainControlAction = [](sts::GameContext&) {};  // no map room to return to (synthetic state)
                end.exitBattle(after);
                results.push_back({{"encounter", name}, {"kind", kind}, {"seed_index", s}, {"seed", seed}, {"variant", v + 1},
                                   {"card", v >= 0 ? card_name(group.candidates[v].card) : "skip"},
                                   {"source", v >= 0 ? group.candidates[v].source : "base"}, {"pre", pre}, {"won", won},
                                   {"start_hp", start_hp}, {"battle_final_hp", won ? static_cast<int>(end.player.curHp) : 0},
                                   {"post", stsrl::game_state::state(after)}, {"simulations", teacher.budget.simulations},
                                   {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count()}});
            }
        }
    }
    return {{"group", group.description}, {"teachers", settings}, {"results", results}};
}

Json play(const Json& input, const stsrl::ValueNet* net) {
    const auto mode = input.at("mode").get<std::string>();
    if (mode == "deck") return {{"group", build_group(input).description}};
    if (mode == "fight") return fight(input, net);
    throw std::invalid_argument{"mode must be deck or fight"};
}

}  // namespace

int main(int argc, char* argv[]) { return stsrl::worker::main(argc, argv, "card_marginals_worker", play); }
