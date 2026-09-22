// Entry-root record generator. Teacher: sts_ml's PublicBeliefCombatSearch with
// guided rollouts (rollout mode 2), built against sts_ml's simulator fork.
// Particle construction mirrors sts_ml bindings' publicBeliefCombatSearchAction.
#include "combat/BattleContext.h"
#include "game/Random.h"
#include "scenarios/slime_entry_projection.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <tuple>

static std::uint64_t combat_seed(std::uint64_t source, std::uint64_t replicate) {
    return source ^ (0x9e3779b97f4a7c15ULL * (replicate + 1));
}

static std::uint64_t next_particle_seed(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    auto value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Battle-only part of sts_ml RLEnvironment::resampleHidden, then resampleDraw.
static sts::BattleContext particle(const sts::BattleContext& observed, std::uint64_t particle_seed) {
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

int main(int argc, char** argv) {
    // Rows follow data/combat/README.md. `max_actions` is the search's maximumActions.
    // Roots are taken with index % worker_count == worker_index; episode ids stay global.
    if (argc != 9) {
        std::cerr << "usage: generate_entry_mcts_records input.jsonl simulations max_actions replicates root_limit random_window worker_index worker_count\n";
        return 2;
    }
    int skipped = 0;
    auto entries = stsrl::scenarios::load_slime_entry_projections(argv[1], skipped);
    const auto simulations = std::stoll(argv[2]);
    const int max_actions = std::stoi(argv[3]);
    const auto replicates = std::stoull(argv[4]);
    // One uniformly random move per fight, at decision U[0, random_window); 0 disables.
    const auto random_window = std::stoull(argv[6]);
    const auto worker_index = std::stoull(argv[7]);
    const auto worker_count = std::stoull(argv[8]);
    constexpr int particles = 8;  // sts_ml agent/config.json boss_particles.
    entries.resize(std::min(entries.size(), static_cast<std::size_t>(std::stoull(argv[5]))));
    for (std::size_t root = worker_index; root < entries.size(); root += worker_count) {
        const auto& entry = entries[root];
        for (std::uint64_t replicate = 0; replicate < replicates; ++replicate) {
        const auto episode = root * replicates + replicate;
        const auto seed = combat_seed(entry.source_seed, replicate);
        std::mt19937_64 explore(seed ^ 0xe9510ULL);  // random-move stream, derived from combat_seed.
        const auto random_at = random_window
            ? std::uniform_int_distribution<std::uint64_t>(0, random_window - 1)(explore) : UINT64_MAX;
        auto env = stsrl::scenarios::slime_entry_projection(entry, seed);
        const auto fight_start = std::chrono::steady_clock::now();
        std::vector<nlohmann::json> rows; int decision = 0, random_moves = 0;
        while (!env.done()) {
            auto d = env.decision();
            const auto& observed = env.battle();
            const auto public_seed = sts::search::PublicBeliefCombatSearch::publicObservation(observed);
            std::vector<sts::BattleContext> states; auto particle_seed = public_seed;
            for (int i = 0; i < particles; ++i) states.push_back(particle(observed, next_particle_seed(particle_seed)));
            sts::search::PublicBeliefCombatSearch search(std::move(states), public_seed, 2);
            search.maximumActions = max_actions;
            search.search(simulations);
            auto legal_index = [&](sts::search::Action action) {
                const auto bits = sts::search::PublicBeliefCombatSearch::mapAction(search.particles.front(), action, observed).bits;
                for (std::size_t i = 0; i < d.legal_actions.size(); ++i) if (env.action_bits(i) == bits) return i;
                throw std::runtime_error("teacher action not in legal set");
            };
            std::size_t chosen = legal_index(search.selectedAction());
            nlohmann::json actions = nlohmann::json::array();
            double value_sum = 0;
            for (const auto& e : search.root().edges) {
                value_sum += e.valueSum;
                if (e.visits == 0) continue;
                const auto i = legal_index(e.action);
                actions.push_back({{"action", i}, {"description", env.action_description(i)},
                                   {"visits", e.visits}, {"mean_value", e.valueSum / e.visits}});
            }
            const auto root_visits = search.root().visits;
            const bool was_random = static_cast<std::uint64_t>(decision) == random_at;
            if (was_random) {
                chosen = std::uniform_int_distribution<std::size_t>(0, d.legal_actions.size() - 1)(explore);
                ++random_moves;
            }
            nlohmann::json row = d.encoding;
            row["episode_id"] = episode; row["decision_index"] = decision++; row["turn"] = observed.turn;
            row["entry_id"] = entry.entry_id; row["deck_signature"] = entry.deck_signature; row["combat_seed"] = seed;
            row["starting_hp"] = entry.hp; row["starting_max_hp"] = entry.max_hp;
            row["actions"] = std::move(actions); row["chosen_action"] = chosen; row["was_random"] = was_random;
            row["root_value"] = root_visits ? value_sum / root_visits : 0.0;
            rows.push_back(std::move(row));
            env.step(chosen);
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - fight_start).count();
        const bool won = env.won();
        const int hp = env.player_hp(), max_hp = env.player_max_hp(), potions = env.battle().potionCount;
        // sts_ml PublicBeliefCombatSearch::scorePrediction with default weights.
        const double terminal_value = won ? (35.0 + hp + 4.0 * potions) / (55.0 + max_hp) : 0.0;
        for (auto& row : rows) {
            row["won"] = won; row["final_hp"] = hp; row["potions"] = potions; row["terminal_value"] = terminal_value;
            auto b = nlohmann::json::to_msgpack(row); std::cout.write((char*)b.data(), b.size());
        }
        std::cout.flush();
        std::cerr << "episode " << episode << " won=" << won << " hp=" << hp << "/" << max_hp << " potions=" << potions
                  << " decisions=" << decision << " random=" << random_moves << " seconds=" << seconds << '\n';
        }
    }
}
