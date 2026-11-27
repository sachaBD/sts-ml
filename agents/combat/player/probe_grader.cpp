#include "agents/combat/player/probe_grader.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace stsrl::player {
namespace {
// Public action keys keep card choices aligned when slot indices differ across worlds.
void play_key(sts::BattleContext& state, std::uint64_t key) {
    for (const auto a : combat::legal_actions(state))
        if (combat::action_key(state, a) == key) { a.execute(state); return; }
    throw std::runtime_error{"probe_grader: action missing in particle"};
}
}  // namespace

std::vector<std::optional<search::Gumbel::Round>> resolve_checkpoints(
    std::span<const search::Gumbel::Round> trace, std::span<const int> budgets, std::optional<int> eval_cap) {
    std::vector<std::optional<search::Gumbel::Round>> out(budgets.size());
    if (trace.empty()) return out;
    for (std::size_t k = 0; k < budgets.size(); ++k) {
        if (eval_cap && budgets[k] > *eval_cap) continue;
        const search::Gumbel::Round* chosen = &trace.front();
        for (const auto& r : trace)
            if (r.evals <= budgets[k]) chosen = &r;
        out[k] = *chosen;
    }
    return out;
}

ProbeGrader::ProbeGrader(const sts::BattleContext& observed, search::Action prior,
                         std::span<const search::Action> candidates, std::uint64_t particle_stream,
                         GradeSettings settings)
    : settings_{settings}, target_turn_{observed.turn + settings.horizon_turns}, moves_{prior} {
    if (settings_.particles < 1 || settings_.horizon_turns < 1 || settings_.max_steps < 1)
        throw std::invalid_argument{"probe_grader: invalid settings"};
    for (const auto c : candidates)
        if (!index(c)) moves_.push_back(c);
    if (moves_.size() == 1) return;  // agreement: zero particles, zero evals

    const auto worlds = combat::particles(observed, settings_.particles, particle_stream);
    branches_.reserve(moves_.size() * std::size_t(settings_.particles));
    scores_.assign(moves_.size() * std::size_t(settings_.particles), 0.0);
    for (const auto move : moves_) {
        const auto key = combat::action_key(observed, move);
        for (const auto& world : worlds) {
            Branch b{world};
            play_key(b.state, key);
            branches_.push_back(std::move(b));
        }
    }
    advance();
}

void ProbeGrader::evaluated(std::span<const search::Evaluation> evaluations) {
    if (pending_.empty() || evaluations.size() != pending_.size())
        throw std::invalid_argument{"probe_grader: one evaluation per pending query"};
    for (std::size_t i = 0; i < evaluations.size(); ++i) {
        const auto& [obs, state, actions] = pending_[i];
        const auto& e = evaluations[i];
        if (e.logits.size() != actions.size()) throw std::invalid_argument{"probe_grader: logit count mismatch"};
        const auto best = std::size_t(std::ranges::max_element(e.logits) - e.logits.begin());
        cache_.emplace(obs, Cached{e.value, combat::action_key(state, actions[best])});
    }
    advance();
}

std::optional<float> ProbeGrader::gain(search::Action action) const {
    if (!queries_.empty()) throw std::logic_error{"probe_grader: unfinished"};
    if (action.bits == moves_.front().bits) return 0.0f;
    const auto m = index(action);
    if (!m) return std::nullopt;
    double sum = 0;
    for (int p = 0; p < settings_.particles; ++p)
        sum += scores_[*m * std::size_t(settings_.particles) + std::size_t(p)] - scores_[std::size_t(p)];
    return float(sum / settings_.particles);
}

std::optional<std::size_t> ProbeGrader::index(search::Action action) const {
    for (std::size_t i = 0; i < moves_.size(); ++i)
        if (moves_[i].bits == action.bits) return i;
    return std::nullopt;
}

void ProbeGrader::advance() {
    pending_.clear();
    queries_.clear();
    std::unordered_set<std::uint64_t> queued;
    for (std::size_t i = 0; i < branches_.size(); ++i) {
        auto& b = branches_[i];
        while (!b.done) {
            if (b.state.outcome != sts::Outcome::UNDECIDED) {
                scores_[i] = b.state.outcome == sts::Outcome::PLAYER_VICTORY;
                b.done = true;
                break;
            }
            const bool stop = b.state.turn >= target_turn_ || b.steps >= settings_.max_steps;
            auto actions = combat::legal_actions(b.state);
            if (!stop && actions.size() == 1) { actions[0].execute(b.state); ++b.steps; continue; }
            const auto obs = combat::observation_key(b.state);
            if (const auto it = cache_.find(obs); it != cache_.end()) {
                if (stop) { scores_[i] = it->second.value; b.done = true; break; }
                play_key(b.state, it->second.action_key);
                ++b.steps;
                continue;
            }
            if (queued.insert(obs).second) pending_.push_back({obs, b.state, std::move(actions)});
            break;
        }
    }
    queries_.reserve(pending_.size());
    for (const auto& p : pending_) queries_.push_back({p.state, p.actions});
}
}  // namespace stsrl::player
