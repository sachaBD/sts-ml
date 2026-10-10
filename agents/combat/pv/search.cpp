#include "agents/combat/pv/search.hpp"

#include "sim/search/PublicBeliefCombatSearch.h"
#include <set>

namespace stsrl::pv {
namespace {
using PublicState = sts::search::PublicBeliefCombatSearch;
}

GuidedRollout::GuidedRollout(const sts::BattleContext& any_state) : searcher_{any_state} {
    searcher_.rolloutMode = 2;
}

double GuidedRollout::operator()(sts::BattleContext& state, std::uint64_t seed) {
    searcher_.randGen.seed(seed);
    sts::search::BattleScumSearcher2::Node scratch;
    for (int step = 0; step < 512 && state.outcome == sts::Outcome::UNDECIDED; ++step) {
        searcher_.rolloutAction(scratch, state).execute(state);
    }
    if (state.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE) {
        throw std::runtime_error{"PV: unsupported simulator effect in a rollout"};
    }
    return state.outcome == sts::Outcome::UNDECIDED ? 0 : CombatObjective::terminal_value(state);
}

std::uint64_t observation_key(const sts::BattleContext& state) {
    return PublicState::observationKey(state);
}

std::uint64_t action_key(const sts::BattleContext& state, sts::search::Action action) {
    return PublicState::publicActionKey(state, action);
}

std::vector<sts::search::Action> legal_actions(const sts::BattleContext& state) {
    std::vector<sts::search::Action> moves;
    std::set<std::uint64_t> keys;

    for (const auto action : sts::search::Action::getAllActionsInState(state)) {
        if (!action.isValidAction(state)) throw std::runtime_error{"PV: enumeration returned an illegal action"};
        if (keys.insert(action_key(state, action)).second) moves.push_back(action);
    }

    if (moves.empty()) throw std::runtime_error{"PV: no legal actions in a search node"};
    return moves;
}

HalvingSchedule halving_schedule(int max_num_considered_actions, std::int64_t num_simulations) {
    // mctx 0.0.71 seq_halving.get_sequence_of_considered_visits, line for line; considered_count records the phase.
    if (num_simulations < 1) throw std::invalid_argument{"PV: halving needs a positive budget"};
    HalvingSchedule out;
    if (max_num_considered_actions <= 1) {
        for (std::int64_t i = 0; i < num_simulations; ++i) { out.considered_visit.push_back(i); out.considered_count.push_back(1); }
        return out;
    }
    const int log2max = int(std::ceil(std::log2(double(max_num_considered_actions))));
    std::vector<std::int64_t> visits(std::size_t(max_num_considered_actions), 0);
    int num_considered = max_num_considered_actions;
    while (std::int64_t(out.considered_visit.size()) < num_simulations) {
        const auto extra = std::max<std::int64_t>(1, std::int64_t(double(num_simulations) / (double(log2max) * num_considered)));
        for (std::int64_t r = 0; r < extra; ++r) {
            for (int i = 0; i < num_considered; ++i) { out.considered_visit.push_back(visits[std::size_t(i)]); out.considered_count.push_back(num_considered); }
            for (int i = 0; i < num_considered; ++i) ++visits[std::size_t(i)];
        }
        num_considered = std::max(2, num_considered / 2);
    }
    out.considered_visit.resize(std::size_t(num_simulations)); out.considered_count.resize(std::size_t(num_simulations));
    return out;
}

std::vector<double> halving_completed_sigma(const std::vector<std::int64_t>& visits, const std::vector<double>& value_sums,
                                            const std::vector<double>& priors, double raw_value,
                                            const SearchSettings::RootHalving& h) {
    const auto n = visits.size();
    if (n == 0 || value_sums.size() != n || priors.size() != n) throw std::invalid_argument{"PV: halving sigma shape"};
    std::int64_t total = 0, most = 0;
    double probs_visited = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total += visits[i]; most = std::max(most, visits[i]);
        if (visits[i] > 0) probs_visited += std::max(priors[i], double(std::numeric_limits<float>::min()));  // mctx finfo(float32).tiny
    }
    double weighted = 0;  // mctx _compute_mixed_value
    for (std::size_t i = 0; i < n; ++i)
        if (visits[i] > 0) weighted += std::max(priors[i], double(std::numeric_limits<float>::min())) * (value_sums[i] / double(visits[i])) / probs_visited;
    const double mixed = (raw_value + double(total) * weighted) / double(total + 1);
    std::vector<double> q(n);
    for (std::size_t i = 0; i < n; ++i) q[i] = visits[i] > 0 ? value_sums[i] / double(visits[i]) : mixed;
    const double lo = *std::min_element(q.begin(), q.end()), hi = *std::max_element(q.begin(), q.end());
    const double scale = (h.maxvisit_init + double(most)) * h.value_scale;
    for (auto& v : q) v = scale * (v - lo) / std::max(hi - lo, 1e-8);
    return q;
}

std::vector<double> policy_priors(const Prediction& prediction, std::size_t action_count) {
    validate_value(prediction.value);
    if (action_count == 0 || prediction.logits.size() != action_count) {
        throw std::invalid_argument{"PV: logits must match the node's legal actions"};
    }
    for (float logit : prediction.logits) {
        if (!std::isfinite(logit)) throw std::invalid_argument{"PV: non-finite policy logit"};
    }

    // Stable softmax only: no uniform floor and no substitution for an invalid network output.
    const double maximum = *std::max_element(prediction.logits.begin(), prediction.logits.end());
    double total = 0;
    std::vector<double> priors;
    for (float logit : prediction.logits) {
        priors.push_back(std::exp(double(logit) - maximum));
        total += priors.back();
    }
    for (auto& prior : priors) prior /= total;
    return priors;
}

void validate_root_state(const sts::BattleContext& state) {
    if (state.outcome != sts::Outcome::UNDECIDED || state.player.cc != sts::CharacterClass::IRONCLAD ||
        state.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE) {
        throw std::invalid_argument{"PV: unsupported root state"};
    }
    // Current particle sampling preserves the current enemy intent. With Dome that intent is hidden,
    // so copying it would give search oracle information. Refuse rather than pretend the belief is valid.
    if (state.player.hasRelic<sts::RelicId::RUNIC_DOME>()) {
        throw std::invalid_argument{"PV search requires visible enemy intents; Rune Dome beliefs are not implemented"};
    }
}

Prediction evaluate_root(Evaluator& evaluator, const sts::BattleContext& state) {
    validate_root_state(state);
    const std::array<Inputs, 1> batch{encode(state, legal_actions(state))};
    auto predictions = evaluator.evaluate(batch);
    if (predictions.size() != 1) throw std::runtime_error{"PV: expected one root prediction"};
    return std::move(predictions.front());
}

void add_root_noise(std::vector<double>& priors, const SearchSettings& settings, std::mt19937_64& random) {
    if (!std::isfinite(settings.noise_fraction) || settings.noise_fraction < 0 || settings.noise_fraction > 1 ||
        !std::isfinite(settings.noise_alpha) || settings.noise_alpha <= 0) {
        throw std::invalid_argument{"PV: invalid root-noise configuration"};
    }
    if (settings.noise_fraction == 0) return;

    std::gamma_distribution<double> gamma{settings.noise_alpha, 1};
    double total = 0;
    std::vector<double> noise;
    for (std::size_t i = 0; i < priors.size(); ++i) {
        noise.push_back(gamma(random));
        total += noise.back();
    }
    if (!std::isfinite(total) || total <= 0) throw std::runtime_error{"PV: invalid Dirichlet draw"};

    for (std::size_t i = 0; i < priors.size(); ++i) {
        priors[i] = (1 - settings.noise_fraction) * priors[i] + settings.noise_fraction * noise[i] / total;
    }
}

}  // namespace stsrl::pv
