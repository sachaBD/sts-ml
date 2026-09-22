// Plays one entry-root Slime fight with the teacher search (PublicBeliefCombatSearch,
// 8 particles, rollout mode 2, objective mode 0), exactly as generate_entry_mcts_records
// does but with no random moves.
//   rollout: plain teacher search.
//   neural:  same search, but every non-terminal leaf is scored by an external value
//            net via requestBatch/submit. Leaves are sent in batches over stdout as
//            framed msgpack {"type":"leaves","states":[encoding...]}; the controller
//            answers with one big-endian float32 per state on stdin.
// The final result is written as a framed msgpack {"type":"result", ...}.
#include "combat/BattleContext.h"
#include "game/Random.h"
#include "scenarios/slime_entry_projection.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"

#include <arpa/inet.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

#include <nlohmann/json.hpp>

namespace {

// Copied from generate_entry_mcts_records.cpp (teacher particle construction).
std::uint64_t next_particle_seed(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    auto value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

sts::BattleContext particle(const sts::BattleContext& observed, std::uint64_t particle_seed) {
    sts::BattleContext bc = observed;
    const auto draw_seed = particle_seed;
    auto next = [&] { return next_particle_seed(particle_seed); };
    const auto sampled_run_seed = next();
    for (int i = 0; i < 14; ++i) next();
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

void write_frame(const nlohmann::json& message) {
    const auto bytes = nlohmann::json::to_msgpack(message);
    const auto length = htonl(static_cast<std::uint32_t>(bytes.size()));
    std::cout.write(reinterpret_cast<const char*>(&length), sizeof(length));
    std::cout.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::cout.flush();
    if (!std::cout) throw std::runtime_error{"failed to write frame"};
}

// Scores every pending leaf with the net, one round trip per batch.
void neural_search(sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations, int batch,
                   std::int64_t& leaf_evaluations) {
    while (search.simulations < simulations) {
        const auto ids = search.requestBatch(batch, simulations, 0, 0);
        if (ids.empty()) continue;  // every simulation in this batch hit a terminal state
        nlohmann::json states = nlohmann::json::array();
        for (const auto id : ids) {
            // Same encoder as env.decision().encoding, applied to the raw leaf state.
            stsrl::CombatEnvironment leaf(search.pending.at(id).state);
            states.push_back(leaf.decision().encoding);
        }
        write_frame({{"type", "leaves"}, {"states", std::move(states)}});
        for (const auto id : ids) {
            std::uint32_t bits{};
            if (!std::cin.read(reinterpret_cast<char*>(&bits), sizeof(bits)))
                throw std::runtime_error{"missing leaf value"};
            const double value = std::clamp(static_cast<double>(std::bit_cast<float>(ntohl(bits))), 0.0, 2.0);
            if (!std::isfinite(value)) throw std::runtime_error{"non-finite leaf value"};
            search.submit(id, value);
        }
        leaf_evaluations += static_cast<std::int64_t>(ids.size());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: play_entry_pbcs source.jsonl deck_signature combat_seed rollout|neural simulations\n";
        return 2;
    }
    int skipped = 0;
    const auto entries = stsrl::scenarios::load_slime_entry_projections(argv[1], skipped);
    const std::string signature = argv[2];
    const auto found = std::find_if(entries.begin(), entries.end(),
                                    [&](const auto& e) { return e.deck_signature == signature; });
    if (found == entries.end()) throw std::invalid_argument{"deck signature not found"};
    const auto seed = std::stoull(argv[3]);
    const std::string mode = argv[4];
    if (mode != "rollout" && mode != "neural") throw std::invalid_argument{"mode must be rollout or neural"};
    const auto simulations = std::stoll(argv[5]);
    constexpr int particles = 8;       // as generate_entry_mcts_records
    constexpr int max_actions = 512;   // as the gen0 data run
    constexpr int batch = 64;

    auto env = stsrl::scenarios::slime_entry_projection(*found, seed);
    const auto start = std::chrono::steady_clock::now();
    int decisions = 0;
    std::int64_t leaf_evaluations = 0;
    while (!env.done()) {
        auto d = env.decision();
        const auto& observed = env.battle();
        const auto public_seed = sts::search::PublicBeliefCombatSearch::publicObservation(observed);
        std::vector<sts::BattleContext> states;
        auto particle_seed = public_seed;
        for (int i = 0; i < particles; ++i) states.push_back(particle(observed, next_particle_seed(particle_seed)));
        sts::search::PublicBeliefCombatSearch search(std::move(states), public_seed, 2);
        search.maximumActions = max_actions;
        if (mode == "rollout") search.search(simulations);
        else neural_search(search, simulations, batch, leaf_evaluations);
        const auto bits = sts::search::PublicBeliefCombatSearch::mapAction(
            search.particles.front(), search.selectedAction(), observed).bits;
        std::size_t chosen = d.legal_actions.size();
        for (std::size_t i = 0; i < d.legal_actions.size(); ++i) if (env.action_bits(i) == bits) chosen = i;
        if (chosen == d.legal_actions.size()) throw std::runtime_error("teacher action not in legal set");
        env.step(chosen);
        ++decisions;
    }
    write_frame({
        {"type", "result"}, {"deck_signature", found->deck_signature}, {"source_seed", found->source_seed},
        {"seed", seed}, {"mode", mode}, {"simulations", simulations}, {"won", env.won()},
        {"final_hp", env.player_hp()}, {"max_hp", env.player_max_hp()}, {"potions", env.battle().potionCount},
        {"decisions", decisions}, {"leaf_evaluations", leaf_evaluations},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
    });
}
