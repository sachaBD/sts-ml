// Plays one entry-root Slime fight with the teacher search (PublicBeliefCombatSearch,
// 8 particles, rollout mode 2, objective mode 0), exactly as bootstrap_fight_worker
// does but with no random moves.
//   rollout: plain teacher search.
//   neural:  same search, but every non-terminal leaf is scored by an external value
//            net via requestBatch/submit. Leaves are sent in batches over stdout as
//            framed msgpack {"type":"leaves","states":[encoding...]}; the controller
//            answers with one big-endian float32 per state on stdin.
//   truncated: like neural, but each new leaf first plays the search's guided rollout for
//            up to <param> turn increments (default 1); a fight that ends there is scored
//            exactly, otherwise the net scores the stopping state.
//   mixed:   like neural, but each leaf value is lambda*V(leaf) + (1-lambda)*score of one
//            full guided rollout from the leaf (AlphaGo 2016); <param> is lambda (default 0.5).
//   --weights <file>: neural/truncated/mixed score leaves in-process with the native C++ value
//            net (models/value_net.hpp) instead of the controller; nothing is read from stdin.
// The final result is written as a framed msgpack {"type":"result", ...}, including a
// wall-clock profile (search, encode, send, rollout, net seconds).
#include "combat/BattleContext.h"
#include "game/Random.h"
#include "scenarios/slime_entry_projection.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"
#include "apps/teacher_budget.hpp"
#include "models/value_net.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <tuple>

#include <nlohmann/json.hpp>

namespace {

// Copied from the bootstrap fight worker (teacher particle construction).
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

// Same as PublicBeliefCombatSearch::boundedRollout + terminalValue (objective mode 0) with
// no turn limit: the search's guided rollout to the end, scored exactly; unfinished scores 0.
double full_rollout_score(sts::search::PublicBeliefCombatSearch& search, sts::BattleContext state) {
    sts::search::BattleScumSearcher2::Node temporary;
    for (int step = 0; step < search.maximumActions && state.outcome == sts::Outcome::UNDECIDED; ++step) {
        temporary.edges.clear();
        search.rollout.enumerateActionsForNode(temporary, state);
        if (temporary.edges.empty()) throw std::runtime_error("empty public rollout support");
        const auto choice = search.rollout.selectRolloutAction(temporary, state);
        temporary.edges[choice].action.execute(state);
    }
    if (state.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE)
        throw std::runtime_error("unsupported effect in mixed rollout");
    if (state.outcome == sts::Outcome::UNDECIDED) return 0.0;
    return std::clamp(sts::search::BattleScumSearcher2::evaluateEndState(state) / search.normalization, -0.1, 2.0);
}

// Scores every pending leaf with the net, one round trip per batch.
//   rollout_turns: guided-rollout turn increments before a leaf is left pending (truncated).
//   lambda: weight on the net value; below 1 it is mixed with one full rollout (mixed).
// Wall-clock split of neural-mode time (seconds).
struct Profile {
    double search = 0, encode = 0, send = 0, rollout = 0;
    double net = 0;  // native forward, or waiting for the controller (unpack, collate, forward)
    std::int64_t batches = 0;
};

double seconds_since(std::chrono::steady_clock::time_point& t) {
    const auto now = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(now - t).count();
    t = now;
    return s;
}

// Scores every pending leaf with the net, one batch at a time: in-process when `native` is
// given, else one round trip per batch to the controller.
//   rollout_turns: guided-rollout turn increments before a leaf is left pending (truncated).
//   lambda: weight on the net value; below 1 it is mixed with one full rollout (mixed).
void neural_search(sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations, int batch,
                   int rollout_turns, double lambda, const stsrl::ValueNet* native,
                   std::int64_t& leaf_evaluations, Profile& profile) {
    std::vector<stsrl::EncodedCombatState> encoded;
    std::vector<float> values;
    while (search.simulations < simulations) {
        auto t = std::chrono::steady_clock::now();
        const auto ids = search.requestBatch(batch, simulations, rollout_turns,
                                             rollout_turns > 0 ? search.maximumActions : 0);
        profile.search += seconds_since(t);
        if (ids.empty()) continue;  // every simulation in this batch hit a terminal state
        encoded.clear();
        for (const auto id : ids) {
            // Same encoder as env.decision().encoding, applied to the raw leaf state.
            stsrl::CombatEnvironment leaf(search.pending.at(id).state);
            encoded.push_back(leaf.decision().encoding);
        }
        profile.encode += seconds_since(t);
        if (!native) write_frame({{"type", "leaves"}, {"states", encoded}});
        profile.send += seconds_since(t);
        std::vector<double> rollout_scores;  // computed while the controller runs the net
        if (lambda < 1.0)
            for (const auto id : ids) rollout_scores.push_back(full_rollout_score(search, search.pending.at(id).state));
        profile.rollout += seconds_since(t);
        values.resize(ids.size());
        if (native) {
            native->evaluate(encoded, values);
        } else {
            for (auto& value : values) {
                std::uint32_t bits{};
                if (!std::cin.read(reinterpret_cast<char*>(&bits), sizeof(bits)))
                    throw std::runtime_error{"missing leaf value"};
                value = std::bit_cast<float>(ntohl(bits));
            }
        }
        profile.net += seconds_since(t);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const double value = std::clamp(static_cast<double>(values[i]), 0.0, 2.0);
            if (!std::isfinite(value)) throw std::runtime_error{"non-finite leaf value"};
            search.submit(ids[i], lambda < 1.0 ? lambda * value + (1.0 - lambda) * rollout_scores[i] : value);
        }
        profile.search += seconds_since(t);
        leaf_evaluations += static_cast<std::int64_t>(ids.size());
        ++profile.batches;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // --weights <file> (last arguments): neural modes score leaves in-process with the native
    // value net instead of asking the controller.
    std::optional<stsrl::ValueNet> native;
    if (argc > 2 && std::string(argv[argc - 2]) == "--weights") {
        native.emplace(argv[argc - 1]);
        argc -= 2;
    }
    // --no-early-stop (last argument): rollout mode spends the full budget on every decision.
    const bool early_stop = !(argc > 1 && std::string(argv[argc - 1]) == "--no-early-stop");
    if (!early_stop) --argc;
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: play_entry_pbcs source.jsonl deck_signature combat_seed rollout|neural|truncated|mixed"
                     " simulations [turns|lambda] [--no-early-stop] [--weights value_weights.bin]\n";
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
    if (mode != "rollout" && mode != "neural" && mode != "truncated" && mode != "mixed")
        throw std::invalid_argument{"mode must be rollout, neural, truncated or mixed"};
    const auto simulations = std::stoll(argv[5]);
    const bool hybrid = mode == "truncated" || mode == "mixed";
    if (argc == 7 && !hybrid) throw std::invalid_argument{"only truncated and mixed take a parameter"};
    const double param = argc == 7 ? std::stod(argv[6]) : mode == "truncated" ? 1.0 : 0.5;
    const int rollout_turns = mode == "truncated" ? static_cast<int>(param) : 0;
    const double lambda = mode == "mixed" ? param : 1.0;
    if (mode == "truncated" && (rollout_turns < 0 || rollout_turns != param))
        throw std::invalid_argument{"turns must be a non-negative integer"};
    if (!(lambda >= 0.0 && lambda <= 1.0)) throw std::invalid_argument{"lambda must be in [0, 1]"};
    constexpr int particles = 8;       // as bootstrap_fight_worker
    constexpr int max_actions = 512;   // as the gen0 data run
    constexpr int batch = 64;
    // Block-stacking decks (e.g. Barricade + Entrench) can stall for hundreds of decisions;
    // a fight still undecided at this turn counts as a loss, flagged as a timeout.
    constexpr int max_turns = 50;

    auto env = stsrl::scenarios::slime_entry_projection(*found, seed);
    const auto start = std::chrono::steady_clock::now();
    int decisions = 0;
    std::int64_t leaf_evaluations = 0, terminal_evaluations = 0, simulations_used = 0;
    bool timeout = false;
    Profile profile;
    while (!env.done()) {
        if (env.battle().turn >= max_turns) { timeout = true; break; }
        auto d = env.decision();
        const auto& observed = env.battle();
        const auto public_seed = sts::search::PublicBeliefCombatSearch::publicObservation(observed);
        std::vector<sts::BattleContext> states;
        auto particle_seed = public_seed;
        for (int i = 0; i < particles; ++i) states.push_back(particle(observed, next_particle_seed(particle_seed)));
        sts::search::PublicBeliefCombatSearch search(std::move(states), public_seed, 2);
        search.maximumActions = max_actions;
        if (mode == "rollout") {
            simulations_used += stsrl::run_teacher_search(search, simulations, d.legal_actions.size(), early_stop);
        } else {
            neural_search(search, simulations, batch, rollout_turns, lambda, native ? &*native : nullptr,
                          leaf_evaluations, profile);
            simulations_used += search.simulations;
        }
        terminal_evaluations += search.terminalEvaluations;
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
        {"seed", seed}, {"mode", mode}, {"simulations", simulations},
        {"param", argc == 7 ? nlohmann::json(argv[6]) : nlohmann::json(nullptr)}, {"won", !timeout && env.won()}, {"timeout", timeout},
        {"turns", env.battle().turn},
        {"final_hp", env.player_hp()}, {"max_hp", env.player_max_hp()}, {"potions", env.battle().potionCount},
        {"decisions", decisions}, {"leaf_evaluations", leaf_evaluations}, {"terminal_evaluations", terminal_evaluations},
        {"simulations_used", simulations_used}, {"native", native.has_value()},
        {"profile", {{"search", profile.search}, {"encode", profile.encode}, {"send", profile.send},
                     {"rollout", profile.rollout}, {"net", profile.net}, {"batches", profile.batches}}},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
    });
}
