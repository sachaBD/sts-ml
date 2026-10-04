#include "agents/combat/pv/turn_search.hpp"
#include "agents/combat/pv/turn_state_key.hpp"
#include "agents/combat/pv/search.hpp"
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace stsrl::pv {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
sts::BattleContext replay(const sts::BattleContext& parent, const TurnChild& child) {
    auto state = parent;
    for (auto bits : child.sequence) {
        const sts::search::Action action{bits};
        if (!action.isValidAction(state)) throw std::runtime_error{"turn search: stored sequence is illegal"};
        action.execute(state);
    }
    if (child.terminal && (state.outcome == sts::Outcome::UNDECIDED ||
        child.key != (state.outcome == sts::Outcome::PLAYER_VICTORY ? "terminal:win" : "terminal:loss")))
        throw std::runtime_error{"turn search: stored terminal outcome changed"};
    if (!child.terminal && turn_state_key(state) != child.key)
        throw std::runtime_error{"turn search: stored sequence changed exact key"};
    return state;
}
std::size_t child_bytes(const TurnChild& child) {
    // Conservative allowance for allocation headers, spare vector/string capacity and DFS dedupe copies.
    return 2 * (sizeof(TurnChild) + child.sequence.capacity() * sizeof(std::uint32_t) + child.key.capacity() + 128);
}
std::size_t tree_bytes(const TurnNode& node) {
    std::size_t bytes = sizeof(TurnNode) + 128;
    for (const auto& child : node.children) bytes += child_bytes(child) + (child.node ? tree_bytes(*child.node) : 0);
    return bytes;
}
void depths(const TurnNode& node, std::size_t depth, std::size_t& count, double& sum, std::size_t& maximum) {
    maximum = std::max(maximum, depth);
    if (node.children.empty()) { ++count; sum += depth; return; }
    for (const auto& child : node.children) {
        if (child.node) depths(*child.node, depth + 1, count, sum, maximum);
        else { ++count; sum += depth + 1; maximum = std::max(maximum, depth + 1); }
    }
}
std::size_t select(const TurnNode& node, double c) {
    double low = 100, high = 0;
    for (const auto& child : node.children) { low = std::min(low, child.q()); high = std::max(high, child.q()); }
    std::size_t best = 0; double score = -1;
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        const auto& child = node.children[i];
        const double q = high > low ? (child.q() - low) / (high - low) : 0;
        const double value = q + c * child.prior * std::sqrt(double(node.visits)) / (1 + child.visits);
        if (value > score) { score = value; best = i; }
    }
    return best;
}
}
TurnSearch::TurnSearch(const sts::BattleContext& state, TurnSearchCaps caps, double c, double temperature)
    : state_{state}, caps_{caps}, c_{c}, temperature_{temperature}, root_{std::make_unique<TurnNode>()} {
    if (!caps.max_sequences || !caps.root_max_children || !caps.max_children || !caps.max_actions || caps.max_actions > 512 ||
        !caps.max_bytes || !(caps.max_seconds > 0) || !std::isfinite(caps.max_seconds) || !std::isfinite(c) || c < 0 ||
        !std::isfinite(temperature) || temperature <= 0)
        throw std::invalid_argument{"turn search: invalid caps/PUCT settings"};
    bytes_ = tree_bytes(*root_);
}

bool TurnSearch::expand(TurnNode& node, const sts::BattleContext& parent, const Evaluate& evaluate) {
    const auto start = Clock::now();
    const auto child_cap = &node == root_.get() ? caps_.root_max_children : caps_.max_children;
    std::string reason;
    auto timed_out = [&] { if (elapsed(start) > caps_.max_seconds) { reason = "seconds"; return true; } return false; };
    std::vector<TurnChild> children;
    std::unordered_set<std::string> seen;
    std::vector<std::uint32_t> path;
    std::size_t sequences = 0, local_bytes = sizeof(TurnNode) + 128, stack_bytes = 0;
    struct Frame { sts::BattleContext state; std::vector<sts::search::Action> moves; std::size_t next = 0, bytes = 0; };
    std::vector<std::unique_ptr<Frame>> stack;
    auto push = [&](const sts::BattleContext& state) {
        // Bound replay/DFS workspace as well as the persistent tree. Dynamic pile storage allowance is conservative.
        const auto cards = state.cards.drawPile.size() + state.cards.discardPile.size() + state.cards.exhaustPile.size();
        const auto frame_bytes = sizeof(Frame) + 4 * cards * sizeof(sts::CardInstance) + 65536;
        if (bytes_ + local_bytes + stack_bytes + frame_bytes > caps_.max_bytes) { reason = "memory"; return; }
        auto frame = std::make_unique<Frame>(); frame->state = state;
        frame->moves = sts::search::Action::getAllActionsInState(state);
        if (frame->moves.empty()) throw std::runtime_error{"turn search: undecided state has no legal actions"};
        frame->bytes = frame_bytes; stack_bytes += frame_bytes;
        stack.push_back(std::move(frame));
    };
    push(parent);
    while (!stack.empty() && reason.empty()) {
        auto& frame = *stack.back();
        if (frame.next == frame.moves.size()) {
            stack_bytes -= frame.bytes; stack.pop_back(); if (!path.empty()) path.pop_back(); continue;
        }
        if (sequences >= caps_.max_sequences) { reason = "sequences"; break; }
        if (timed_out()) break;
        if (path.size() >= caps_.max_actions) { reason = "actions"; break; }
        auto action = frame.moves[frame.next++];
        auto next = frame.state;
        if (!action.isValidAction(next)) throw std::runtime_error{"turn search: enumerated illegal action"};
        path.push_back(action.bits); action.execute(next);
        if (next.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE)
            throw std::runtime_error{"turn search: unsupported simulator effect"};
        if (timed_out()) break;
        const bool terminal = next.outcome != sts::Outcome::UNDECIDED;
        // END_TURN may pause for mandatory choices. Complete those before keying the next turn.
        if (terminal || (next.turn != parent.turn && next.inputState == sts::InputState::PLAYER_NORMAL)) {
            ++sequences;
            const bool won = next.outcome == sts::Outcome::PLAYER_VICTORY;
            const auto key = terminal ? std::string(won ? "terminal:win" : "terminal:loss") : turn_state_key(next);
            if (seen.insert(key).second) {
                if (children.size() >= child_cap) { reason = "children"; break; }
                TurnChild child; child.sequence = path; child.key = key; child.terminal = terminal;
                child.value = terminal && won ? 100 : 0;
                local_bytes += child_bytes(child);
                if (bytes_ + local_bytes + stack_bytes > caps_.max_bytes) { reason = "memory"; break; }
                children.push_back(std::move(child));
            }
            path.pop_back();
        } else push(next);
    }
    stack.clear(); seen.clear();
    // All-or-nothing expansion: never expose children before exhaustive enumeration AND evaluation succeed.
    if (reason.empty()) {
        constexpr std::size_t batch_size = 32;
        for (std::size_t begin = 0; begin < children.size() && reason.empty();) {
            if (timed_out()) break;
            std::vector<Inputs> inputs; std::vector<std::size_t> indices;
            std::size_t input_bytes = 0;
            while (begin < children.size() && inputs.size() < batch_size) {
                const auto i = begin++;
                if (children[i].terminal) continue;
                auto next = replay(parent, children[i]);
                auto encoded = encode(next, legal_actions(next));
                for (const auto& rows : encoded) input_bytes += rows.capacity() * sizeof(float);
                if (bytes_ + local_bytes + 4 * input_bytes > caps_.max_bytes) { reason = "memory"; break; }
                indices.push_back(i); inputs.push_back(std::move(encoded));
                if (timed_out()) break;
            }
            if (!reason.empty()) break;
            if (!inputs.empty()) {
                ++stats_.network_calls; stats_.evaluated_states += inputs.size();
                const auto predictions = evaluate(inputs);
                if (predictions.size() != inputs.size()) throw std::runtime_error{"turn search: evaluator batch mismatch"};
                for (std::size_t j = 0; j < predictions.size(); ++j) {
                    if (!std::isfinite(predictions[j].value)) throw std::runtime_error{"turn search: nonfinite value"};
                    children[indices[j]].value = std::clamp(double(predictions[j].value), 0.0, 100.0);
                }
                if (timed_out()) break;
            }
        }
    }
    if (elapsed(start) > caps_.max_seconds) {
        stats_.time_overshoot = true;
        if (reason.empty()) reason = "seconds";
    }
    if (!reason.empty()) { node.capped = true; node.reason = reason; return false; }
    if (children.empty()) throw std::runtime_error{"turn search: exhaustive expansion produced no children"};
    double best = 0, total = 0;
    for (const auto& child : children) best = std::max(best, child.value);
    for (auto& child : children) { child.prior = std::exp((child.value - best) / temperature_); total += child.prior; }
    for (auto& child : children) child.prior /= total;
    if (timed_out()) { stats_.time_overshoot = true; node.capped = true; node.reason = reason; return false; }
    node.children = std::move(children); node.expanded = true; node.capped = false; node.reason.clear(); node.value = best;
    bytes_ += local_bytes;
    return true;
}

std::vector<std::uint32_t> TurnSearch::decide(std::size_t budget, const Evaluate& evaluate, double noise_fraction,
                                              std::mt19937_64* random, bool sample) {
    if ((noise_fraction != 0 || sample) && !random)
        throw std::invalid_argument{"turn search: noise/sampling requires RNG"};
    if (!budget) throw std::invalid_argument{"turn search: budget must be positive"};
    const auto start = Clock::now(); stats_ = {}; stats_.reused = reused_;
    std::size_t used = 0;
    if (!root_->expanded) {
        ++used; ++stats_.expansions;
        if (!expand(*root_, state_, evaluate)) {
            stats_.fallback = true; stats_.reason = root_->reason; stats_.seconds = elapsed(start); return {};
        }
        ++root_->visits;
    }
    if (noise_fraction != 0 && !root_noise_done_) {
        std::vector<double> priors;
        for (const auto& child : root_->children) priors.push_back(child.prior);
        SearchSettings settings; settings.noise_fraction = noise_fraction;
        add_root_noise(priors, settings, *random);
        for (std::size_t i=0; i<priors.size(); ++i) root_->children[i].prior = priors[i];
        root_noise_done_ = true;
    }
    while (used++ < budget) {
        ++stats_.expansions;
        TurnNode* node = root_.get(); auto state = state_;
        std::vector<TurnChild*> path; std::vector<TurnNode*> nodes{node};
        double value = node->value;
        while (node->expanded) {
            auto& child = node->children[select(*node, c_)]; path.push_back(&child);
            value = child.value;
            if (child.terminal || child.capped) break;
            state = replay(state, child);
            if (!child.node) {
                if (bytes_ + sizeof(TurnNode) + 128 > caps_.max_bytes) { child.capped = true; break; }
                child.node = std::make_unique<TurnNode>(); child.node->value = child.value; bytes_ += sizeof(TurnNode) + 128;
            }
            node = child.node.get(); nodes.push_back(node);
            if (node->capped) { value = node->value; break; }
            if (!node->expanded) { expand(*node, state, evaluate); value = node->value; break; }
        }
        for (auto* child : path) { ++child->visits; child->sum += value; }
        for (auto* visited : nodes) ++visited->visits;
    }
    selected_ = 0;
    for (std::size_t i = 1; i < root_->children.size(); ++i) {
        const auto& child = root_->children[i]; const auto& best = root_->children[selected_];
        if (child.visits > best.visits || (child.visits == best.visits && child.q() > best.q())) selected_ = i;
    }
    if (sample) {
        std::vector<double> weights;
        for (const auto& child : root_->children) weights.push_back(child.visits);
        if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0)
            throw std::runtime_error{"turn search: sampling has no visited children"};
        selected_ = std::discrete_distribution<std::size_t>(weights.begin(), weights.end())(*random);
    }
    stats_.root_children = root_->children.size();
    std::size_t leaves = 0; double sum = 0; depths(*root_, 0, leaves, sum, stats_.max_depth);
    stats_.mean_leaf_depth = leaves ? sum / leaves : 0; stats_.seconds = elapsed(start);
    return root_->children[selected_].sequence;
}

bool TurnSearch::advance(const sts::BattleContext& actual) {
    if (stats_.fallback || actual.outcome != sts::Outcome::UNDECIDED || root_->children.empty()) return false;
    auto& chosen = root_->children[selected_];
    if (chosen.terminal || turn_state_key(actual) != chosen.key) return false;
    auto next = std::move(chosen.node);
    reused_ = bool(next);
    if (!next) { next = std::make_unique<TurnNode>(); next->value = chosen.value; }
    root_ = std::move(next); root_noise_done_ = false; state_ = actual; bytes_ = tree_bytes(*root_);
    if (bytes_ > caps_.max_bytes) { root_ = std::make_unique<TurnNode>(); bytes_ = tree_bytes(*root_); reused_ = false; }
    return true;
}
} // namespace stsrl::pv
