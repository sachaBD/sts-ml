// One seeded Ironclad run. SimpleAgent plays to the Act-1 Slime Boss;
// public-belief search teaches the boss fight. Output: OUTPUT_DIR/fight.msgpack.
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
#include <optional>
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

std::optional<sts::BattleContext> reach_slime_boss(std::uint64_t seed, int ascension) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, ascension};
    if (game.boss != sts::MonsterEncounter::SLIME_BOSS) return std::nullopt;  // act 1 boss is fixed at game creation
    sts::search::SimpleAgent agent;
    agent.curGameContext = &game;
    while (game.outcome == sts::GameOutcome::UNDECIDED && game.act == 1) {
        if (game.screenState != sts::ScreenState::BATTLE) {
            agent.stepOutOfCombat(game);
            continue;
        }
        sts::BattleContext battle;
        battle.init(game);
        if (game.curRoom == sts::Room::BOSS) {
            if (battle.encounter == sts::MonsterEncounter::SLIME_BOSS) return battle;
            return std::nullopt;
        }
        agent.playoutBattle(battle);
        battle.exitBattle(game);
    }
    return std::nullopt;
}

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

void set_identity(Json& row, std::uint64_t seed, int decision, int turn, int starting_hp, int max_hp) {
    row["episode_id"] = static_cast<std::int64_t>(seed);
    row["decision_index"] = decision;
    row["turn"] = turn;
    row["entry_id"] = nullptr; // This is a full run, not a saved boss-entry projection.
    row["deck_signature"] = nullptr;
    row["combat_seed"] = seed;
    row["starting_hp"] = starting_hp;
    row["starting_max_hp"] = max_hp;
}

void record_children(std::vector<Json>& rows, const stsrl::CombatEnvironment& env,
                     const SearchDecision& decision, std::uint64_t seed, int index, int hp, int max_hp) {
    for (const auto& tried : decision.tried) {
        if (tried.index == decision.chosen || tried.visits < child_min_visits) continue;
        stsrl::CombatEnvironment child{env.battle()};
        (void)child.decision();
        child.step(tried.index);
        if (child.done()) continue;
        Json row = child.decision().encoding;
        set_identity(row, seed, index, child.battle().turn, hp, max_hp);
        row["actions"] = Json::array();
        row["chosen_action"] = -1;
        row["was_random"] = false;
        row["root_value"] = tried.value;
        row["row_kind"] = "child";
        row["parent_action"] = tried.index;
        row["simulations_used"] = 0;
        rows.push_back(std::move(row));
    }
}

Json play_boss(sts::BattleContext battle, std::uint64_t seed) {
    stsrl::CombatEnvironment env{std::move(battle)};
    const int starting_hp = env.player_hp(), max_hp = env.player_max_hp();
    std::mt19937_64 rng(seed ^ 0xe9510ULL);
    const int random_at = std::uniform_int_distribution<int>{0, random_window - 1}(rng);
    std::vector<Json> rows, children;
    int index = 0;
    while (!env.done()) {
        auto state = env.decision();
        auto choice = search_decision(env, state.legal_actions.size());
        const bool random = index == random_at;
        if (random) choice.chosen = std::uniform_int_distribution<std::size_t>{0, state.legal_actions.size() - 1}(rng);
        record_children(children, env, choice, seed, index, starting_hp, max_hp);
        Json row = state.encoding;
        set_identity(row, seed, index, env.battle().turn, starting_hp, max_hp);
        row.update({{"actions", std::move(choice.actions)}, {"chosen_action", choice.chosen},
                    {"was_random", random}, {"root_value", choice.value}, {"row_kind", "decision"},
                    {"parent_action", -1}, {"simulations_used", choice.used}});
        rows.push_back(std::move(row));
        env.step(choice.chosen);
        ++index;
    }
    for (auto& child : children) rows.push_back(std::move(child));
    const int hp = env.player_hp(), potions = env.battle().potionCount;
    const double terminal = env.won() ? (35.0 + hp + 4.0 * potions) / (55.0 + max_hp) : 0.0;
    for (auto& row : rows) row.update({{"won", env.won()}, {"final_hp", hp},
                                       {"potions", potions}, {"terminal_value", terminal}});
    return {{"seed", seed}, {"status", "boss_fight"}, {"won", env.won()}, {"rows", rows}};
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
        const auto boss = reach_slime_boss(seed, ascension);
        const Json result = boss ? play_boss(*boss, seed)
                                 : Json{{"seed", seed}, {"status", "no_boss"}, {"rows", Json::array()}};
        write_result(directory, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fight worker (seed " << argv[1] << "): " << error.what() << '\n';
        return 1;
    }
}
