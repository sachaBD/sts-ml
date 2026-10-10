// pv_continuation: same-root continuations for diagnostics (experiments/single-deck-expert-iteration/CONTINUATION.md).
//
// One JSON request per stdin line, one JSON response per stdout line:
//   {"start": <combat_v4 start>, "ops": [action bits...], "mode": "learner" | "teacher", "sims": N,
//    "particles": [i, ...], "true_state": false, "record": false}   ("roots": fingerprints only, no play)
// The state is rebuilt as BattleContext::init(start_game(start)) followed by each legal action (as champ_session).
// Roots: particle i of stsrl::teacher::public_particles(state, max(particles)+1), exactly as champ_session's
// `playout`; with "true_state" the actual state itself (gates only). Each root's opaque fingerprint (seed, all RNG
// states, every pile in order, player/monster fields) is reported for equivalence checks only.
//   learner: the frozen pv_worker `play` loop without exploration (fresh non-oracle tree per decision,
//            root evaluated once, 8 public particles, default PUCT / settings, selected_action, forced moves
//            unsearched) and its
//            guards: turn < 50 (request "turn_cap": test-only override of 50) and total actions (prefix included) < 512 -> otherwise status "capped".
//   teacher: champ_session `playout` (guided rollout, 8 particles, fresh search per decision, search salt i + 1);
//            no turn cap (as the recorded teacher labels).
// Outcome: the actual PLAYER_VICTORY flag. MODEL.onnx (argv[1]) is required for learner mode.
// Phase3 additions (all optional; absent fields keep the Stage1/Stage2 semantics exactly):
//   "chain": [{"particle": i, "ops": [...]}, ...]  after start+ops, replace the state by public particle i of it and
//            apply ops (legality-checked; illegal = hard error). Reconstructs a state inside an episode's own sampled world.
//   "force": bits  applied inside every sampled root particle BEFORE the learner continuation (draws/end turn use the
//            particle's RNG). Illegal in a particle, or a particle legal menu differing from the state's = hard error.
//   mode "query": no play; at the state: learner 2k fresh-tree choice/visits/root (same code as a learner step),
//            teacher guided-rollout choice/visits (public particles, no oracle, salt 0), and pv::encode features
//            (features only; no outcome label).
//   "perturb": true  (tests only) permute the state's hidden draw order and replace every RNG/seed before use.
//   record mode adds a per-decision "trace" (draw size, input state, select task) for known-top inference.
//   Every root also reports "root_semantic": the fingerprint without card uniqueIds.
#include "agents/combat/pv/search.hpp"
#include "agents/combat/pv/objective.hpp"
#include "apps/pv/shard.hpp"
#include "agents/combat/search/teacher_search.hpp"
#include "environments/combat/record_v4.hpp"
#include "sim/search/Action.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
namespace pv = stsrl::pv;
using Clock = std::chrono::steady_clock;

namespace {

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

struct Hasher {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    void add(std::uint64_t v) { for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xff; h *= 0x100000001b3ULL; } }
    void card(const sts::CardInstance& c) {
        add(static_cast<int>(c.id)); add(c.upgraded); add(c.cost); add(c.costForTurn); add(static_cast<std::uint16_t>(c.uniqueId));
        add(static_cast<std::uint16_t>(c.specialData)); add(c.freeToPlayOnce); add(c.retain);
    }
};

// Opaque equivalence fingerprint (never a model feature).
std::uint64_t fingerprint(const sts::BattleContext& s) {
    Hasher h;
    h.add(s.seed);
    for (const auto* r : {&s.aiRng, &s.cardRandomRng, &s.miscRng, &s.monsterHpRng, &s.potionRng, &s.shuffleRng}) {
        h.add(static_cast<std::uint32_t>(r->counter)); h.add(r->seed0); h.add(r->seed1);
    }
    h.add(s.turn); h.add(static_cast<int>(s.inputState)); h.add(static_cast<int>(s.outcome));
    h.add(s.player.curHp); h.add(s.player.energy); h.add(s.player.block); h.add(s.player.strength);
    for (int i = 0; i < s.cards.cardsInHand; ++i) h.card(s.cards.hand[i]);
    h.add(0xd1); for (const auto& c : s.cards.drawPile) h.card(c);
    h.add(0xd2); for (const auto& c : s.cards.discardPile) h.card(c);
    h.add(0xd3); for (const auto& c : s.cards.exhaustPile) h.card(c);
    for (int i = 0; i < s.monsters.monsterCount; ++i) {
        const auto& m = s.monsters.arr[i];
        h.add(static_cast<int>(m.id)); h.add(m.curHp); h.add(m.block); h.add(static_cast<int>(m.moveHistory[0]));
        h.add(static_cast<int>(m.moveHistory[1])); h.add(m.miscInfo); h.add(m.strength);
    }
    return h.h;
}

// Same, without card uniqueIds (not features; the sampler's canonical sort does not order equal-key cards by them).
std::uint64_t semantic_fingerprint(sts::BattleContext s) {
    for (auto* pile : {&s.cards.drawPile, &s.cards.discardPile, &s.cards.exhaustPile}) for (auto& c : *pile) c.uniqueId = 0;
    for (int i = 0; i < s.cards.cardsInHand; ++i) s.cards.hand[i].uniqueId = 0;
    return fingerprint(s);
}

std::string hex(std::uint64_t v) { char b[17]; std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v)); return b; }

std::uint64_t rng_fingerprint(const sts::Random& r) {
    Hasher h; h.add(static_cast<std::uint32_t>(r.counter)); h.add(r.seed0); h.add(r.seed1); return h.h;
}

sts::BattleContext build(const Json& request) {
    sts::BattleContext bc;
    bc.init(stsrl::combat_v4::start_game(request.at("start")));
    for (const auto& op : request.value("ops", Json::array())) {
        const sts::search::Action action{op.get<std::uint32_t>()};
        if (bc.outcome != sts::Outcome::UNDECIDED || !action.isValidAction(bc)) throw std::runtime_error{"illegal prefix action"};
        action.execute(bc);
    }
    for (const auto& link : request.value("chain", Json::array())) {
        const int i = link.at("particle").get<int>();
        if (i < 0 || bc.outcome != sts::Outcome::UNDECIDED) throw std::runtime_error{"bad chain link"};
        bc = stsrl::teacher::public_particles(bc, i + 1)[i];
        for (const auto& op : link.at("ops")) {
            const sts::search::Action action{op.get<std::uint32_t>()};
            if (bc.outcome != sts::Outcome::UNDECIDED || !action.isValidAction(bc)) throw std::runtime_error{"illegal chained action"};
            action.execute(bc);
        }
    }
    if (request.value("perturb", false)) {  // tests only: hidden draw order and every RNG/seed replaced
        std::uint64_t x = 0x5eed5eedULL;
        const auto next = [&x] { x = x * 6364136223846793005ULL + 1442695040888963407ULL; return x >> 11; };
        auto& d = bc.cards.drawPile;
        for (int i = static_cast<int>(d.size()) - 1; i > 0; --i) std::swap(d[i], d[next() % (i + 1)]);
        bc.seed = next();
        for (auto* r : {&bc.aiRng, &bc.cardRandomRng, &bc.miscRng, &bc.monsterHpRng, &bc.potionRng, &bc.shuffleRng}) *r = sts::Random(next());
    }
    return bc;
}

Json query(const sts::BattleContext& state, pv::Evaluator& evaluator, std::int64_t sims, std::int64_t teacher_sims) {
    const auto moves = pv::legal_actions(state);
    Json out{{"legal_actions", Json::array()}};
    for (const auto& m : moves) out["legal_actions"].push_back(m.bits);
    const auto inputs = pv::encode(state, moves);
    Json features = Json::array();
    for (const auto& v : inputs) features.push_back(v);
    out["features"] = std::move(features);
    if (moves.size() < 2) throw std::runtime_error{"query needs a decision with >1 legal action"};
    {   // learner: identical to one decision of learner() below
        pv::SearchSettings settings;
        const double c = pv::PuctScore{}.exploration;
        const auto root_prediction = pv::evaluate_root(evaluator, state);
        auto particles = stsrl::teacher::public_particles(state, 8);
        pv::Search<> tree{state, std::move(particles), root_prediction, settings, pv::PuctScore{c}};
        tree.run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, sims);
        double visits = 0, value_sum = 0; Json children = Json::array();
        for (const auto& edge : tree.root().edges) {
            visits += edge.visits; value_sum += edge.value_sum;
            if (edge.visits) children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
        }
        out["learner"] = {{"action", tree.selected_action().bits}, {"root_value", value_sum / visits}, {"children", children}, {"sims", sims}};
    }
    {   // teacher: the continuation teacher searcher, one fresh public search (no oracle, salt 0)
        const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {teacher_sims, stsrl::teacher::particles});
        stsrl::teacher::tweaks().search_salt = 0;
        stsrl::CombatEnvironment env{state};
        const auto legal = env.legal_action_count();
        const auto d = stsrl::teacher::search_decision(env, legal, searcher, false, stsrl::teacher::particles);
        Json children = Json::array();
        for (const auto& a : d.actions) children.push_back({{"action", env.action_bits(a.at("action").get<std::size_t>())},
                                                             {"visits", a.at("visits")}, {"value", a.at("mean_value")}});
        out["teacher"] = {{"action", env.action_bits(d.chosen)}, {"root_value", d.value}, {"children", children}, {"sims", teacher_sims}, {"used", d.used}};
    }
    return out;
}

// Frozen pv_worker `play` (explore off, sample_turns off, oracle off, policy_only off), from `state`.
Json learner(sts::BattleContext state, pv::Evaluator& evaluator, std::int64_t sims, std::size_t prior_actions, bool record,
             const std::string& shard_path = {}, const std::string& fight_id = {}, int encounter = 39, int turn_cap = 50) {
    pv::SearchSettings settings;  // noise_fraction 0, rollout_mix 0, oracle false (worker defaults)
    const double c = pv::PuctScore{}.exploration;
    struct Pending { pv::Inputs inputs; std::vector<std::uint32_t> moves; std::vector<float> policy; std::optional<float> root; int step, turn; bool has_policy; };
    std::vector<Pending> pending;
    Json search = Json::array(), trace = Json::array(); std::vector<std::uint32_t> actions;
    while (state.outcome == sts::Outcome::UNDECIDED && state.turn < turn_cap && prior_actions + actions.size() < 512) {
        const auto moves = pv::legal_actions(state);
        auto chosen = moves.front();
        std::optional<float> root; std::vector<float> policy(moves.size(), 0); bool has_policy = false;
        if (moves.size() > 1) {
            const auto root_prediction = pv::evaluate_root(evaluator, state);
            auto particles = stsrl::teacher::public_particles(state, 8);
            pv::Search<> tree{state, std::move(particles), root_prediction, settings, pv::PuctScore{c}};
            tree.run([&evaluator](std::span<const pv::Inputs> batch) { return evaluator.evaluate(batch); }, sims);
            chosen = tree.selected_action();
            double visits = 0, value_sum = 0;
            for (const auto& edge : tree.root().edges) { visits += edge.visits; value_sum += edge.value_sum; }
            for (std::size_t i = 0; i < moves.size(); ++i) for (const auto& edge : tree.root().edges) if (edge.action.bits == moves[i].bits) policy[i] = float(edge.visits / visits);
            root = float(value_sum / visits); has_policy = true;
            if (record) {
                Json children = Json::array();
                for (const auto& edge : tree.root().edges) if (edge.visits) {
                    children.push_back({{"action", edge.action.bits}, {"visits", edge.visits}, {"value", edge.value_sum / edge.visits}});
                }
                search.push_back({{"step", prior_actions + actions.size()}, {"root_value", value_sum / visits}, {"children", children}});
            }
        }
        if (record) trace.push_back({{"step", prior_actions + actions.size()}, {"turn", state.turn}, {"draw", state.cards.drawPile.size()},
                                     {"input", static_cast<int>(state.inputState)}, {"task", static_cast<int>(state.cardSelectInfo.cardSelectTask)},
                                     {"legal", moves.size()}});
        if (!shard_path.empty()) {
            std::vector<std::uint32_t> bits; for (const auto& m : moves) bits.push_back(m.bits);
            pending.push_back({pv::encode(state, moves), std::move(bits), std::move(policy), root, int(prior_actions + actions.size()), state.turn, has_policy});
        }
        actions.push_back(chosen.bits); chosen.execute(state);
    }
    const bool done = state.outcome != sts::Outcome::UNDECIDED;
    if (!shard_path.empty() && done) {
        const auto tmp = shard_path + ".tmp"; pv::ShardWriter writer{tmp};
        const float target = float(pv::CombatObjective::terminal_value(state));
        for (const auto& x : pending) writer.add({fight_id, state.seed, encounter, x.step, x.turn,
            state.outcome == sts::Outcome::PLAYER_VICTORY, state.player.curHp, target, x.root, x.moves, x.policy, x.has_policy, &x.inputs});
        writer.close(); std::filesystem::rename(tmp, shard_path);
    }
    Json out{{"status", done ? "completed" : "capped"}, {"won", state.outcome == sts::Outcome::PLAYER_VICTORY},
             {"hp", state.player.curHp}, {"turns", state.turn}, {"continuation_actions", actions.size()}};
    if (record) { out["actions"] = actions; out["search"] = search; out["trace"] = trace; }
    return out;
}

Json teacher(const sts::BattleContext& root, std::int64_t sims, int particle) {
    const auto searcher = stsrl::teacher::leaf_search({"guided_rollout", 0, 0}, nullptr, {sims, stsrl::teacher::particles});
    stsrl::teacher::tweaks().search_salt = static_cast<std::uint64_t>(particle) + 1;
    const auto end = stsrl::teacher::play_fight(root, searcher, false, stsrl::teacher::particles, false);
    stsrl::teacher::tweaks().search_salt = 0;
    const bool completed = end.outcome != sts::Outcome::UNDECIDED;
    return {{"status", completed ? "completed" : "error"}, {"won", end.outcome == sts::Outcome::PLAYER_VICTORY}, {"hp", end.player.curHp}, {"turns", end.turn}};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view{argv[1]} == "--self-test") {
        const auto make_rng = [](std::int32_t counter, std::uint64_t seed0, std::uint64_t seed1) {
            sts::Random r; r.counter = counter; r.seed0 = seed0; r.seed1 = seed1; return r;
        };
        const auto a = make_rng(7, 11, 13), same = make_rng(7, 11, 13);
        const auto counter_changed = make_rng(8, 11, 13), seed_changed = make_rng(7, 11, 14);
        const auto ha = rng_fingerprint(a);
        const bool ok = ha == rng_fingerprint(same) && ha != rng_fingerprint(counter_changed) && ha != rng_fingerprint(seed_changed);
        std::cout << Json{{"rng_fingerprint_deterministic", ok}, {"fingerprint", hex(ha)}}.dump() << std::endl;
        return ok ? 0 : 1;
    }
    std::unique_ptr<pv::Evaluator> evaluator;
    if (argc > 1) evaluator = std::make_unique<pv::Evaluator>(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Json out;
        try {
            const auto request = Json::parse(line);
            const auto state = build(request);
            if (state.outcome != sts::Outcome::UNDECIDED) throw std::runtime_error{"the fight is over"};
            const auto mode = request.at("mode").get<std::string>();
            const auto sims = request.at("sims").get<std::int64_t>();
            const bool record = request.value("record", false);
            const auto prefix = request.value("ops", Json::array()).size();
            out["true_fingerprint"] = hex(fingerprint(state));
            Json legal = Json::array();
            for (const auto& action : pv::legal_actions(state)) legal.push_back(action.bits);
            out["legal_actions"] = std::move(legal);
            if (mode == "query") {
                if (!evaluator) throw std::runtime_error{"query mode needs MODEL.onnx"};
                out["query"] = query(state, *evaluator, sims, request.value("teacher_sims", std::int64_t{20000}));
                out["true_semantic"] = hex(semantic_fingerprint(state));
                std::cout << out.dump() << std::endl;
                continue;
            }
            std::vector<std::pair<int, sts::BattleContext>> roots;
            if (request.value("true_state", false)) roots.push_back({-1, state});
            else {
                const auto ids = request.at("particles").get<std::vector<int>>();
                int most = 0; for (int i : ids) { if (i < 0) throw std::runtime_error{"negative particle"}; most = std::max(most, i); }
                const auto particles = stsrl::teacher::public_particles(state, most + 1);
                for (int i : ids) roots.push_back({i, particles[i]});
            }
            Json results = Json::array();
            const bool forcing = request.contains("force");
            for (auto& [i, root] : roots) {
                const auto t = Clock::now();
                Json r = Json::object();
                const auto root_fp = fingerprint(root), root_sem = semantic_fingerprint(root);
                std::size_t prefix_here = prefix;
                if (forcing) {
                    std::vector<std::uint32_t> menu; for (const auto& a : pv::legal_actions(root)) menu.push_back(a.bits);
                    if (Json(menu) != out["legal_actions"]) throw std::runtime_error{"particle legal menu differs from the state's"};
                    const sts::search::Action forced{request.at("force").get<std::uint32_t>()};
                    if (std::find(menu.begin(), menu.end(), forced.bits) == menu.end() || !forced.isValidAction(root))
                        throw std::runtime_error{"forced action illegal in particle"};
                    forced.execute(root); ++prefix_here;
                    r["forced"] = forced.bits; r["after_force_fingerprint"] = hex(fingerprint(root));
                    r["after_force_hand"] = Json::array();
                    for (int h = 0; h < root.cards.cardsInHand; ++h) r["after_force_hand"].push_back(static_cast<int>(root.cards.hand[h].id));
                }
                if (mode == "roots") {}  // fingerprints only, no play
                else if (mode == "learner") {
                    if (!evaluator) throw std::runtime_error{"learner mode needs MODEL.onnx"};
                    if (forcing && root.outcome != sts::Outcome::UNDECIDED) {  // the forced action ended the fight
                        r.update({{"status", "completed"}, {"won", root.outcome == sts::Outcome::PLAYER_VICTORY}, {"hp", root.player.curHp},
                                  {"turns", root.turn}, {"continuation_actions", 0}});
                    } else r.update(learner(root, *evaluator, sims, prefix_here, record, request.value("shard_path", std::string{}),
                                request.value("fight_id", std::string{}), request.at("start").value("encounter", 39),
                                request.value("turn_cap", 50)));  // test-only override; default = frozen worker cap
                } else if (mode == "teacher") r.update(teacher(root, sims, i));
                else throw std::runtime_error{"unknown mode " + mode};
                r["particle"] = i; r["root_fingerprint"] = hex(root_fp); r["root_semantic"] = hex(root_sem); r["seconds"] = since(t);
                results.push_back(std::move(r));
            }
            out["results"] = std::move(results);
        } catch (const std::exception& e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
