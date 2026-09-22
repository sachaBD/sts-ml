// Bootstrap fight worker. Plays one Act-1 Slime Boss fight per job, with sts_ml's
// PublicBeliefCombatSearch as the teacher (guided rollouts, rollout mode 2).
//
// Jobs arrive on stdin, one JSON object per line:
//   {"episode_id": 7, "combat_seed": 123, "entry": {<accepted slime entry row>}}
// Closing stdin is the graceful stop: the current fight finishes, then the worker exits 0.
//
// stdout: one msgpack object per finished fight, {"episode_id", "won", "rows": [...]},
//         rows as described by runs/README.md (schema combat_v1).
// stderr: progress lines, "hb ..." per decision and "fight ..." per finished fight.
#include "combat/BattleContext.h"
#include "game/Random.h"
#include "scenarios/slime_entry_projection.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"
#include "apps/teacher_budget.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <tuple>

namespace {

struct Options {
    std::int64_t simulations = 15000;
    int max_actions = 512;
    std::uint64_t random_window = 24;  // one random move per fight at decision U[0, N); 0 disables
    std::int64_t child_min_visits = 50;  // child rows for non-chosen root moves; 0 disables
    bool early_stop = true;
    int particles = 8;  // sts_ml agent/config.json boss_particles
};

std::uint64_t next_particle_seed(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    auto value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Battle-only part of sts_ml RLEnvironment::resampleHidden, then resampleDraw.
sts::BattleContext particle(const sts::BattleContext& observed, std::uint64_t particle_seed) {
    sts::BattleContext bc = observed;
    const auto draw_seed = particle_seed;
    auto next = [&] { return next_particle_seed(particle_seed); };
    const auto sampled_run_seed = next();
    for (int i = 0; i < 14; ++i) next();  // GameContext RNG draws, kept for seed-stream parity.
    bc.seed = sampled_run_seed;
    bc.aiRng = sts::Random(next());
    bc.cardRandomRng = sts::Random(next());
    bc.miscRng = sts::Random(next());
    bc.monsterHpRng = sts::Random(next());
    bc.potionRng = sts::Random(next());
    bc.shuffleRng = sts::Random(next());
    if (bc.inputState != sts::InputState::CARD_SELECT) {
        std::sort(bc.cards.drawPile.begin(), bc.cards.drawPile.end(), [](const auto& l, const auto& r) {
            return std::tie(l.id, l.upgraded, l.specialData, l.cost, l.costForTurn, l.uniqueId)
                 < std::tie(r.id, r.upgraded, r.specialData, r.cost, r.costForTurn, r.uniqueId);
        });
        java::Collections::shuffle(bc.cards.drawPile.begin(), bc.cards.drawPile.end(), java::Random(next()));
    }
    sts::search::PublicBeliefCombatSearch::resampleDraw(bc, observed, draw_seed);
    return bc;
}

void play_fight(const Options& options, std::uint64_t episode, std::uint64_t seed,
                const stsrl::scenarios::SlimeEntryProjection& entry) {
    std::mt19937_64 explore(seed ^ 0xe9510ULL);  // random-move stream, derived from combat_seed
    const auto random_at = options.random_window
        ? std::uniform_int_distribution<std::uint64_t>(0, options.random_window - 1)(explore) : UINT64_MAX;
    auto env = stsrl::scenarios::slime_entry_projection(entry, seed);
    const auto fight_start = std::chrono::steady_clock::now();
    std::vector<nlohmann::json> rows, child_rows;
    int decision = 0, random_moves = 0;
    std::int64_t fight_simulations = 0;
    while (!env.done()) {
        const auto decision_start = std::chrono::steady_clock::now();
        auto d = env.decision();
        const auto& observed = env.battle();
        const auto public_seed = sts::search::PublicBeliefCombatSearch::publicObservation(observed);
        std::vector<sts::BattleContext> states;
        auto particle_seed = public_seed;
        for (int i = 0; i < options.particles; ++i) states.push_back(particle(observed, next_particle_seed(particle_seed)));
        sts::search::PublicBeliefCombatSearch search(std::move(states), public_seed, 2);
        search.maximumActions = options.max_actions;
        const auto used = stsrl::run_teacher_search(search, options.simulations, d.legal_actions.size(), options.early_stop);
        fight_simulations += used;
        auto legal_index = [&](sts::search::Action action) {
            const auto bits = sts::search::PublicBeliefCombatSearch::mapAction(search.particles.front(), action, observed).bits;
            for (std::size_t i = 0; i < d.legal_actions.size(); ++i) if (env.action_bits(i) == bits) return i;
            throw std::runtime_error("teacher action not in legal set");
        };
        std::size_t chosen = legal_index(search.selectedAction());
        nlohmann::json actions = nlohmann::json::array();
        double value_sum = 0;
        struct Tried { std::size_t index; std::int64_t visits; double q; };
        std::vector<Tried> tried;
        for (const auto& e : search.root().edges) {
            value_sum += e.valueSum;
            if (e.visits == 0) continue;
            const auto i = legal_index(e.action);
            actions.push_back({{"action", i}, {"description", env.action_description(i)},
                               {"visits", e.visits}, {"mean_value", e.valueSum / e.visits}});
            tried.push_back({i, static_cast<std::int64_t>(e.visits), e.valueSum / e.visits});
        }
        const auto root_visits = search.root().visits;
        const bool was_random = static_cast<std::uint64_t>(decision) == random_at;
        if (was_random) {
            chosen = std::uniform_int_distribution<std::size_t>(0, d.legal_actions.size() - 1)(explore);
            ++random_moves;
        }
        nlohmann::json row = d.encoding;
        row["episode_id"] = episode; row["decision_index"] = decision; row["turn"] = observed.turn;
        row["entry_id"] = entry.entry_id; row["deck_signature"] = entry.deck_signature; row["combat_seed"] = seed;
        row["starting_hp"] = entry.hp; row["starting_max_hp"] = entry.max_hp;
        row["actions"] = std::move(actions); row["chosen_action"] = chosen; row["was_random"] = was_random;
        row["root_value"] = root_visits ? value_sum / root_visits : 0.0;
        row["row_kind"] = "decision"; row["parent_action"] = -1; row["simulations_used"] = used;
        // Child rows: positions search asks about but the teacher did not play, labelled by the
        // teacher's mean value for that move. Applied to the true state; only the public encoding is kept.
        for (const auto& t : tried) {
            if (options.child_min_visits <= 0 || t.index == chosen || t.visits < options.child_min_visits) continue;
            stsrl::CombatEnvironment child{observed};
            (void)child.decision();
            child.step(t.index);
            if (child.done()) continue;  // terminal values are exact in search; no label needed
            nlohmann::json c = child.decision().encoding;
            c["episode_id"] = episode; c["decision_index"] = decision; c["turn"] = child.battle().turn;
            c["entry_id"] = entry.entry_id; c["deck_signature"] = entry.deck_signature; c["combat_seed"] = seed;
            c["starting_hp"] = entry.hp; c["starting_max_hp"] = entry.max_hp;
            c["actions"] = nlohmann::json::array(); c["chosen_action"] = -1; c["was_random"] = false;
            c["root_value"] = t.q; c["row_kind"] = "child"; c["parent_action"] = t.index;
            c["simulations_used"] = 0;
            child_rows.push_back(std::move(c));
        }
        rows.push_back(std::move(row));
        env.step(chosen);
        std::cerr << "hb episode=" << episode << " decision=" << decision << " turn=" << observed.turn
                  << " sims=" << used << " secs="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - decision_start).count()
                  << std::endl;
        ++decision;
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fight_start).count();
    const bool won = env.won();
    const int hp = env.player_hp(), max_hp = env.player_max_hp(), potions = env.battle().potionCount;
    // sts_ml PublicBeliefCombatSearch::scorePrediction with default weights.
    const double terminal_value = won ? (35.0 + hp + 4.0 * potions) / (55.0 + max_hp) : 0.0;
    for (auto& c : child_rows) rows.push_back(std::move(c));  // after the fight's decision rows
    nlohmann::json fight = {{"episode_id", episode}, {"won", won}, {"rows", nlohmann::json::array()}};
    for (auto& row : rows) {
        row["won"] = won; row["final_hp"] = hp; row["potions"] = potions; row["terminal_value"] = terminal_value;
        fight["rows"].push_back(std::move(row));
    }
    const auto packed = nlohmann::json::to_msgpack(fight);
    std::cout.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    std::cout.flush();
    std::cerr << "fight episode=" << episode << " won=" << won << " hp=" << hp << "/" << max_hp
              << " potions=" << potions << " decisions=" << decision << " rows=" << rows.size()
              << " sims=" << fight_simulations << " random=" << random_moves << " secs=" << seconds << std::endl;
}

std::int64_t integer_argument(int argc, char** argv, int index, const char* flag) {
    if (index >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
    return std::stoll(argv[index]);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--simulations") options.simulations = integer_argument(argc, argv, ++i, "--simulations");
        else if (flag == "--max-actions") options.max_actions = static_cast<int>(integer_argument(argc, argv, ++i, "--max-actions"));
        else if (flag == "--random-window") options.random_window = static_cast<std::uint64_t>(integer_argument(argc, argv, ++i, "--random-window"));
        else if (flag == "--child-min-visits") options.child_min_visits = integer_argument(argc, argv, ++i, "--child-min-visits");
        else if (flag == "--particles") options.particles = static_cast<int>(integer_argument(argc, argv, ++i, "--particles"));
        else if (flag == "--no-early-stop") options.early_stop = false;
        else {
            std::cerr << "usage: bootstrap_fight_worker [--simulations N] [--max-actions N] [--random-window N]"
                         " [--child-min-visits N] [--particles N] [--no-early-stop]   (jobs on stdin)\n";
            return 2;
        }
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const auto job = nlohmann::json::parse(line);
        play_fight(options,
                   job.at("episode_id").get<std::uint64_t>(),
                   job.at("combat_seed").get<std::uint64_t>(),
                   stsrl::scenarios::parse_slime_entry_projection(job.at("entry").dump()));
    }
    return 0;
}
