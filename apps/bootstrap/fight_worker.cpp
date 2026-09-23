// One seeded Ironclad act 1 (combat_v3, see apps/bootstrap/schema.py). Runs whose act 1 boss isn't
// Slime Boss stop at game creation. SimpleAgent plays everything out of combat; public-belief search
// teaches (plays and records) every combat until death or the boss is beaten.
// Output: OUTPUT_DIR/fight.msgpack = {seed, status, floor, fights, rows}.
#include "apps/teacher_budget.hpp"
#include "combat/environment.hpp"
#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "game/Random.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cctype>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

constexpr std::int64_t simulations = 15'000;
constexpr int particles = 8;
constexpr int max_actions = 512;
constexpr std::int64_t child_min_visits = 50;
constexpr int random_window = 24;

std::uint64_t next_seed(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    auto value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

sts::BattleContext sample_particle(const sts::BattleContext& observed, std::uint64_t seed) {
    sts::BattleContext sampled = observed;
    const auto draw_seed = seed;
    const auto next = [&] { return next_seed(seed); };
    sampled.seed = next();
    for (int i = 0; i < 14; ++i) next();
    sampled.aiRng = sts::Random(next());
    sampled.cardRandomRng = sts::Random(next());
    sampled.miscRng = sts::Random(next());
    sampled.monsterHpRng = sts::Random(next());
    sampled.potionRng = sts::Random(next());
    sampled.shuffleRng = sts::Random(next());
    if (sampled.inputState != sts::InputState::CARD_SELECT) {
        std::sort(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), [](const auto& a, const auto& b) {
            return std::tie(a.id, a.upgraded, a.specialData, a.cost, a.costForTurn, a.uniqueId)
                 < std::tie(b.id, b.upgraded, b.specialData, b.cost, b.costForTurn, b.uniqueId);
        });
        java::Collections::shuffle(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), java::Random(next()));
    }
    sts::search::PublicBeliefCombatSearch::resampleDraw(sampled, observed, draw_seed);
    return sampled;
}

sts::search::PublicBeliefCombatSearch make_search(const sts::BattleContext& observed) {
    const auto public_seed = sts::search::PublicBeliefCombatSearch::publicObservation(observed);
    auto stream = public_seed;
    std::vector<sts::BattleContext> states;
    states.reserve(particles);
    for (int i = 0; i < particles; ++i) states.push_back(sample_particle(observed, next_seed(stream)));
    sts::search::PublicBeliefCombatSearch search{std::move(states), public_seed, 2};
    search.maximumActions = max_actions;
    return search;
}

std::size_t legal_index(const stsrl::CombatEnvironment& env, std::size_t count,
                        const sts::search::PublicBeliefCombatSearch& search, sts::search::Action action) {
    const auto bits = sts::search::PublicBeliefCombatSearch::mapAction(
        search.particles.front(), action, env.battle()).bits;
    for (std::size_t i = 0; i < count; ++i)
        if (env.action_bits(i) == bits) return i;
    throw std::runtime_error{"search action is not legal in the real battle"};
}

struct TriedAction {
    std::size_t index;
    std::int64_t visits;
    double value;
};

struct SearchDecision {
    Json actions = Json::array();
    std::vector<TriedAction> tried;
    std::size_t chosen;
    double value;
    std::int64_t used;
};

SearchDecision search_decision(const stsrl::CombatEnvironment& env, std::size_t legal_count) {
    auto search = make_search(env.battle());
    SearchDecision result;
    result.used = stsrl::run_teacher_search(search, simulations, legal_count, true);
    result.chosen = legal_index(env, legal_count, search, search.selectedAction());
    double value_sum = 0;
    for (const auto& edge : search.root().edges) {
        value_sum += edge.valueSum;
        if (!edge.visits) continue;
        const auto index = legal_index(env, legal_count, search, edge.action);
        const auto mean = edge.valueSum / edge.visits;
        result.actions.push_back({{"action", index}, {"description", env.action_description(index)},
                                  {"visits", edge.visits}, {"mean_value", mean}});
        result.tried.push_back({index, static_cast<std::int64_t>(edge.visits), mean});
    }
    result.value = search.root().visits ? value_sum / search.root().visits : 0.0;
    return result;
}

// easy / hard: act 1 hallway pools; event: a fight started by an event ("?" room).
std::string category(const sts::GameContext& game, sts::MonsterEncounter encounter) {
    namespace pool = sts::MonsterEncounterPool;
    if (game.curRoom == sts::Room::BOSS) return "boss";
    if (game.curRoom == sts::Room::ELITE) return "elite";
    if (game.curRoom == sts::Room::MONSTER) {
        const auto act = game.act - 1;
        const auto in = [&](const auto* list, int count) { return std::find(list, list + count, encounter) != list + count; };
        if (in(pool::weakEnemies[act], pool::weakCount[act])) return "easy";
        if (in(pool::strongEnemies[act], pool::strongCount[act])) return "hard";
    }
    return "event";
}

std::string encounter_name(sts::MonsterEncounter encounter) {
    std::string name = sts::monsterEncounterEnumNames[static_cast<int>(encounter)];
    for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return name;
}

// Columns shared by every row of one fight.
Json fight_columns(const sts::GameContext& game, const sts::BattleContext& battle, std::uint64_t seed,
                   int fight_index) {
    return {{"run_seed", seed}, {"episode_id", static_cast<std::int64_t>(seed) * 100 + fight_index},
            {"fight_index", fight_index}, {"act", game.act}, {"floor", game.floorNum},
            {"encounter", encounter_name(battle.encounter)}, {"category", category(game, battle.encounter)},
            {"ascension", game.ascension}, {"starting_hp", battle.player.curHp},
            {"starting_max_hp", battle.player.maxHp}};
}

void record_children(std::vector<Json>& rows, const stsrl::CombatEnvironment& env,
                     const SearchDecision& decision, const Json& fight, int index) {
    for (const auto& tried : decision.tried) {
        if (tried.index == decision.chosen || tried.visits < child_min_visits) continue;
        stsrl::CombatEnvironment child{env.battle()};
        (void)child.decision();
        child.step(tried.index);
        if (child.done()) continue;
        Json row = child.decision().encoding;
        row.update(fight);
        row.update({{"decision_index", index}, {"turn", child.battle().turn}, {"actions", Json::array()},
                    {"chosen_action", -1}, {"was_random", false}, {"root_value", tried.value},
                    {"row_kind", "child"}, {"parent_action", tried.index}, {"simulations_used", 0}});
        rows.push_back(std::move(row));
    }
}

// Teacher plays one fight; its rows are appended to `rows`. Returns the finished battle.
sts::BattleContext play_fight(sts::BattleContext battle, const Json& fight, std::vector<Json>& rows) {
    stsrl::CombatEnvironment env{std::move(battle)};
    const int max_hp = env.player_max_hp();
    std::mt19937_64 rng(fight.at("episode_id").get<std::uint64_t>() ^ 0xe9510ULL);
    const int random_at = std::uniform_int_distribution<int>{0, random_window - 1}(rng);
    std::vector<Json> decisions, children;
    int index = 0;
    while (!env.done()) {
        auto state = env.decision();
        auto choice = search_decision(env, state.legal_actions.size());
        const bool random = index == random_at;
        if (random) choice.chosen = std::uniform_int_distribution<std::size_t>{0, state.legal_actions.size() - 1}(rng);
        record_children(children, env, choice, fight, index);
        Json row = state.encoding;
        row.update(fight);
        row.update({{"decision_index", index}, {"turn", env.battle().turn}, {"actions", std::move(choice.actions)},
                    {"chosen_action", choice.chosen}, {"was_random", random}, {"root_value", choice.value},
                    {"row_kind", "decision"}, {"parent_action", -1}, {"simulations_used", choice.used}});
        decisions.push_back(std::move(row));
        env.step(choice.chosen);
        ++index;
    }
    const int hp = env.player_hp(), potions = env.battle().potionCount;
    const double terminal = env.won() ? (35.0 + hp + 4.0 * potions) / (55.0 + max_hp) : 0.0;
    const Json outcome = {{"won", env.won()}, {"final_hp", hp}, {"potions", potions}, {"terminal_value", terminal}};
    for (auto* part : {&decisions, &children})
        for (auto& row : *part) {
            row.update(outcome);
            rows.push_back(std::move(row));
        }
    return env.battle();
}

Json play_run(std::uint64_t seed, int ascension) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    if (game.boss != sts::MonsterEncounter::SLIME_BOSS)  // act 1 boss is fixed at game creation
        return {{"seed", seed}, {"status", "other_boss"}, {"floor", 0}, {"fights", 0}, {"rows", Json::array()}};
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    std::vector<Json> rows;
    int fights = 0;
    bool boss_beaten = false;
    while (game.outcome == sts::GameOutcome::UNDECIDED && game.act == 1 && !boss_beaten) {
        if (game.screenState != sts::ScreenState::BATTLE) {
            agent.stepOutOfCombat(game);
            continue;
        }
        sts::BattleContext battle;
        battle.init(game);
        const auto end = play_fight(battle, fight_columns(game, battle, seed, fights++), rows);
        end.exitBattle(game);
        boss_beaten = game.curRoom == sts::Room::BOSS && game.outcome == sts::GameOutcome::UNDECIDED;
    }
    const auto status = boss_beaten ? "act_complete" : game.outcome == sts::GameOutcome::PLAYER_LOSS ? "died" : "stopped";
    return {{"seed", seed}, {"status", status}, {"floor", game.floorNum}, {"fights", fights}, {"rows", rows}};
}

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

void write_result(const fs::path& directory, const Json& result) {
    const auto bytes = Json::to_msgpack(result);
    const auto path = directory / "fight.msgpack";
    std::ofstream output{path, std::ios::binary};
    if (!output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error{"failed to write " + path.string()};
    output.close();
    if (!output) throw std::runtime_error{"failed to close " + path.string()};
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: bootstrap_fight_worker SEED ASCENSION OUTPUT_DIR\n";
        return 2;
    }
    try {
        const auto seed = parse_seed(argv[1]);
        const auto ascension = parse_ascension(argv[2]);
        const fs::path directory{argv[3]};
        fs::create_directories(directory);
        write_result(directory, play_run(seed, ascension));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fight worker (seed " << argv[1] << "): " << error.what() << '\n';
        return 1;
    }
}
