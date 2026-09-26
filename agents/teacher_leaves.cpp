#include "agents/teacher_leaves.hpp"

#include "game/Random.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace stsrl::teacher {
namespace {

using Json = nlohmann::json;
using sts::search::PublicBeliefCombatSearch;

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
    PublicBeliefCombatSearch::resampleDraw(sampled, observed, draw_seed);
    return sampled;
}

// Most-visited root edge can no longer be caught with `left` simulations. Never under max backup:
// the played move is the best-valued edge, which the visit gap says nothing about.
bool decided(const PublicBeliefCombatSearch& search, std::int64_t left) {
    if (search.maxBackup) return false;
    std::int64_t best = 0, second = 0;
    for (const auto& e : search.root().edges) {
        if (e.visits > best) { second = best; best = e.visits; }
        else if (e.visits > second) second = e.visits;
    }
    return best - second > left;
}

std::vector<sts::BattleContext> root_particles(const sts::BattleContext& observed, bool oracle, int particles) {
    if (particles < 1) throw std::invalid_argument{"particles must be >= 1"};
    auto stream = PublicBeliefCombatSearch::publicObservation(observed);
    std::vector<sts::BattleContext> states;
    states.reserve(particles);
    if (oracle) states.push_back(observed);
    else for (int i = 0; i < particles; ++i) states.push_back(sample_particle(observed, next_seed(stream)));
    return states;
}

}  // namespace

PublicBeliefCombatSearch make_search(const sts::BattleContext& observed, bool oracle, int particles) {
    const auto public_seed = PublicBeliefCombatSearch::publicObservation(observed);
    PublicBeliefCombatSearch search{root_particles(observed, oracle, particles), public_seed, 2};
    search.maximumActions = max_actions;
    search.maxBackup = oracle;
    return search;
}

std::int64_t run_teacher_search(PublicBeliefCombatSearch& search, std::int64_t simulations,
                                std::size_t legal_moves, bool early_stop,
                                std::int64_t forced_simulations, std::int64_t chunk) {
    if (!early_stop) { search.search(simulations); return simulations; }
    if (legal_moves == 1) {
        const auto n = std::min(forced_simulations, simulations);
        search.search(n);
        return n;
    }
    std::int64_t used = 0;
    while (used < simulations) {
        const auto n = std::min(chunk, simulations - used);
        search.search(n);
        used += n;
        if (decided(search, simulations - used)) break;
    }
    return used;
}

void rebase_search(PublicBeliefCombatSearch& search, const sts::BattleContext& before, std::uint32_t played_bits,
                   const sts::BattleContext& after, bool oracle, int particles) {
    const auto key = PublicBeliefCombatSearch::publicActionKey(before, sts::search::Action{played_bits});
    search.rebase(root_particles(after, oracle, particles), key, PublicBeliefCombatSearch::publicObservation(after));
}

LeafEvaluator value_net_evaluator(const ValueNet& net) {
    return [&net, encoded = std::vector<EncodedCombatState>{}](const std::vector<const sts::BattleContext*>& leaves,
                                                             std::vector<float>& values) mutable {
        encoded.clear();
        for (const auto* leaf : leaves) encoded.push_back(CombatEnvironment{*leaf}.decision().encoding);
        net.evaluate(encoded, values);
    };
}

std::int64_t run_leaf_search(PublicBeliefCombatSearch& search, const LeafEvaluator& evaluate,
                             std::int64_t simulations, std::size_t legal_moves, int rollout_turns,
                             int rollout_steps) {
    if (rollout_turns < 0 || rollout_steps < 0) throw std::invalid_argument{"negative rollout bound"};
    const auto budget = legal_moves == 1 ? std::min(forced_simulations, simulations) : simulations;
    std::vector<const sts::BattleContext*> leaves;
    std::vector<float> values;
    while (search.simulations < budget) {
        const auto ids = search.requestBatch(value_net_batch, budget, rollout_turns, rollout_steps);
        if (!ids.empty()) {  // empty: every simulation in this batch ended the fight
            leaves.clear();
            for (const auto id : ids) leaves.push_back(&search.pending.at(id).state);
            values.clear();
            evaluate(leaves, values);
            if (values.size() != ids.size()) throw std::runtime_error{"leaf evaluator returned the wrong count"};
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (!std::isfinite(values[i])) throw std::runtime_error{"non-finite leaf value"};
                search.submit(ids[i], std::clamp(static_cast<double>(values[i]), 0.0, 2.0));
            }
        }
        if (legal_moves > 1 && decided(search, budget - search.simulations)) break;
    }
    return search.simulations;
}

std::int64_t run_value_net_search(PublicBeliefCombatSearch& search, const ValueNet& net,
                                  std::int64_t simulations, std::size_t legal_moves) {
    return run_leaf_search(search, value_net_evaluator(net), simulations, legal_moves);
}

std::size_t legal_index(const CombatEnvironment& env, std::size_t count, const PublicBeliefCombatSearch& search,
                        sts::search::Action action) {
    const auto bits = PublicBeliefCombatSearch::mapAction(search.particles.front(), action, env.battle()).bits;
    for (std::size_t i = 0; i < count; ++i)
        if (env.action_bits(i) == bits) return i;
    throw std::runtime_error{"search action is not legal in the real battle"};
}

void validate(const Leaf& leaf, bool has_net) {
    const bool bounded = leaf.rollout_turns != 0 || leaf.rollout_steps != 0;
    if (leaf.kind == "guided_rollout" || leaf.kind == "value_net") {
        if (bounded) throw std::invalid_argument{leaf.kind + " takes no rollout bounds"};
        if (has_net != (leaf.kind == "value_net"))
            throw std::invalid_argument{leaf.kind + (has_net ? " takes no value net" : " needs a value net")};
    } else if (leaf.kind == "hybrid") {
        if (leaf.rollout_turns < 1 || leaf.rollout_steps < 1)
            throw std::invalid_argument{"hybrid needs rollout_turns >= 1 and rollout_steps >= 1"};
        if (!has_net) throw std::invalid_argument{"hybrid needs a value net"};
    } else {
        throw std::invalid_argument{"unknown leaf: " + leaf.kind};
    }
}

SearchFn leaf_search(const Leaf& leaf, const ValueNet* net, const Budget& budget) {
    validate(leaf, net != nullptr);
    if (budget.simulations < 1 || budget.particles < 1)
        throw std::invalid_argument{"simulations and particles must be >= 1"};
    if (leaf.kind == "guided_rollout") return guided_rollout_search(budget.simulations);
    if (leaf.kind == "value_net") return value_net_search(*net, budget.simulations);
    return hybrid_search(*net, leaf.rollout_turns, leaf.rollout_steps, budget.simulations);
}

SearchFn guided_rollout_search(std::int64_t simulations) {
    return [simulations](PublicBeliefCombatSearch& search, std::size_t legal_moves) {
        return run_teacher_search(search, simulations, legal_moves, true);
    };
}

SearchFn value_net_search(const ValueNet& net, std::int64_t simulations) {
    return [&net, simulations](PublicBeliefCombatSearch& search, std::size_t legal_moves) {
        return run_value_net_search(search, net, simulations, legal_moves);
    };
}

SearchFn hybrid_search(const ValueNet& net, int rollout_turns, int rollout_steps, std::int64_t simulations) {
    validate({"hybrid", rollout_turns, rollout_steps}, true);
    return [&net, rollout_turns, rollout_steps, simulations](PublicBeliefCombatSearch& search, std::size_t legal_moves) {
        return run_leaf_search(search, value_net_evaluator(net), simulations, legal_moves, rollout_turns,
                               rollout_steps);
    };
}

Json search_settings(const Leaf& leaf, const Budget& budget) {
    Json result = {{"leaf", leaf.kind}, {"particles", budget.particles}, {"simulations", budget.simulations},
                   {"early_stop", true},
                   {"forced_simulations", forced_simulations}, {"max_actions", max_actions}};
    if (leaf.kind == "guided_rollout") result["chunk"] = chunk;
    else result["batch"] = value_net_batch;
    if (leaf.kind == "hybrid") {
        result["rollout_turns"] = leaf.rollout_turns;
        result["rollout_steps"] = leaf.rollout_steps;
    }
    return result;
}

}  // namespace stsrl::teacher
