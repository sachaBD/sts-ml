#pragma once

#include "agents/combat/pv/evaluator.hpp"
#include "agents/combat/pv/objective.hpp"
#include "agents/combat/pv/selection.hpp"

#include "sim/search/BattleScumSearcher2.h"

#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>

namespace stsrl::pv {

struct SearchSettings {
    bool oracle = false;
    int batch_size = 32;
    int maximum_actions = 512;
    double noise_alpha = 0.3;
    double noise_fraction = 0;  // evaluation: off; self-play can explicitly set 0.25
    // AlphaGo-style leaf mixing: backed-up value = (1 - rollout_mix) * V_net + rollout_mix * guided rollout.
    // 0 (default) = network only: no rollout is run and the search is unchanged.
    double rollout_mix = 0;
    // Opt-in root-only Sequential Halving with Gumbel (mctx 0.0.71 gumbel_muzero_root_action_selection,
    // get_sequence_of_considered_visits, score_considered, qtransform_completed_by_mix_value and the
    // gumbel_muzero_policy recommendation). Interior nodes keep PUCT; this is not full Gumbel MuZero.
    // Disabled (default): the search is unchanged.
    struct RootHalving {
        bool enabled = false;
        int max_considered = 16;
        double gumbel_scale = 0.0;     // 0 (default): deterministic Sequential Halving on logits + sigma(completed Q); no RNG draws
        double value_scale = 0.1;      // mctx qtransform_completed_by_mix_value defaults
        double maxvisit_init = 50.0;
    } root_halving;
};

// mctx seq_halving.get_sequence_of_considered_visits, plus the considered-action count of each entry (its phase).
struct HalvingSchedule {
    std::vector<std::int64_t> considered_visit;
    std::vector<int> considered_count;
};
HalvingSchedule halving_schedule(int max_num_considered_actions, std::int64_t num_simulations);
// mctx qtransform_completed_by_mix_value (rescale_values, use_mixed_value) for one node's children: completed Q
// (unvisited: mixed value of raw_value and prior-weighted visited Q), min-max rescaled (eps 1e-8), times
// (maxvisit_init + max visits) * value_scale. value_sums / visits are completed backups.
std::vector<double> halving_completed_sigma(const std::vector<std::int64_t>& visits, const std::vector<double>& value_sums,
                                            const std::vector<double>& priors, double raw_value,
                                            const SearchSettings::RootHalving& settings);

// The teacher's guided rollout (BattleScumSearcher2 rollout mode 2, as PublicBeliefCombatSearch uses it).
class GuidedRollout {
public:
    explicit GuidedRollout(const sts::BattleContext& any_state);
    // Plays `state` to its end and returns CombatObjective::terminal_value; deterministic given `seed`.
    // A fight still undecided after 512 actions is unresolved evidence, never a victory: value 0, as the teacher scores it.
    double operator()(sts::BattleContext& state, std::uint64_t seed);
private:
    sts::search::BattleScumSearcher2 searcher_;
};

// The old search class is used only for public-observation/action utilities, never as a search driver.
std::vector<sts::search::Action> legal_actions(const sts::BattleContext&);
std::uint64_t observation_key(const sts::BattleContext&);
std::uint64_t action_key(const sts::BattleContext&, sts::search::Action);
std::vector<double> policy_priors(const Prediction&, std::size_t action_count);
void validate_root_state(const sts::BattleContext&);
Prediction evaluate_root(Evaluator&, const sts::BattleContext&);
void add_root_noise(std::vector<double>& priors, const SearchSettings&, std::mt19937_64&);

// Dedicated PV tree. Construction requires a root prediction; every traversable node has evaluated priors.
// Score and Selection are independent compile-time policies. Default: AlphaZero-style PUCT + argmax.
template<class Score = PuctScore, class Selection = ArgmaxSelection>
class Search {
public:
    struct Node;
    struct Edge {
        sts::search::Action action;
        std::uint64_t key;
        double prior = 0;
        std::int64_t visits = 0;
        double value_sum = 0;
        int in_flight = 0;
        std::map<std::uint64_t, std::unique_ptr<Node>> outcomes;
    };

    struct Node {
        std::vector<Edge> edges;
        std::optional<double> value;  // absent until this node's network evaluation completes
        std::int64_t visits = 0;
        int in_flight = 0;
    };

    Search(const sts::BattleContext& observed, std::vector<sts::BattleContext> particles,
           const Prediction& root_prediction, SearchSettings settings = {}, Score score = {}, Selection select = {})
        : particles_{std::move(particles)}, settings_{settings}, score_{std::move(score)},
          select_{std::move(select)}, random_{observation_key(observed)}, root_{make_root(observed)} {
        if (settings_.oracle && particles_.size() != 1) throw std::invalid_argument{"PV oracle needs one true state"};
        if (particles_.empty() || settings_.batch_size < 1 || settings_.maximum_actions < 1) {
            throw std::invalid_argument{"PV: invalid particles or search bounds"};
        }
        if (!(settings_.rollout_mix >= 0 && settings_.rollout_mix <= 1)) {
            throw std::invalid_argument{"PV: rollout_mix must be in [0, 1]"};
        }
        if (settings_.rollout_mix > 0) rollout_.emplace(particles_.front());
        for (const auto& particle : particles_) {
            if (observation_key(particle) != observation_key(observed)) {
                throw std::invalid_argument{"PV: particle does not match the root observation"};
            }
            (void)ordered_actions(root_, particle);
        }

        evaluate_node(root_, root_prediction);
        std::vector<double> priors;
        for (const auto& edge : root_.edges) priors.push_back(edge.prior);
        add_root_noise(priors, settings_, random_);
        for (std::size_t i = 0; i < priors.size(); ++i) root_.edges[i].prior = priors[i];
        if (settings_.root_halving.enabled) init_root_halving(observed, root_prediction);
    }

    // Per-decision depth telemetry, over this tree's simulations. A path's depth is its number of actions
    // (terminal action included); turns counts the END_TURN actions on it, i.e. player turns crossed.
    struct Telemetry {
        std::int64_t depth_sum = 0, turns_sum = 0, nodes = 1;  // nodes: the root plus every node created
        int depth_max = 0, turns_max = 0;
        std::vector<std::int64_t> turns_hist;
        std::int64_t batches = 0, evaluated_leaves = 0, phase_flushes = 0;  // network calls, leaves in them, halving elimination batch ends
    };
    const Telemetry& telemetry() const { return telemetry_; }
    const Node& root() const { return root_; }
    std::int64_t simulations() const { return simulations_; }

    sts::search::Action selected_action() const {
        if (root_.in_flight || simulations_ == 0) throw std::logic_error{"PV: search is incomplete"};
        if (settings_.root_halving.enabled) {
            // gumbel_muzero_policy: the best of the most-visited actions by gumbel + logits + sigma(completed Q).
            std::int64_t most = 0;
            for (const auto& edge : root_.edges) most = std::max(most, edge.visits);
            return root_.edges[halving_argmax(most, false)].action;
        }
        const auto best = std::max_element(root_.edges.begin(), root_.edges.end(),
            [](const Edge& a, const Edge& b) { return a.visits < b.visits; });
        return best->action;  // played move: most visits; separate from traversal selection
    }

    // Root halving only: softmax(logits + sigma(completed Q)) over the root's legal actions (mctx action_weights).
    std::vector<double> improved_policy() const {
        if (!settings_.root_halving.enabled) throw std::logic_error{"PV: improved_policy needs root halving"};
        const auto sigma = completed_sigma();
        std::vector<double> z(root_.edges.size());
        for (std::size_t i = 0; i < z.size(); ++i) z[i] = root_logits_[i] + sigma[i];
        const double top = *std::max_element(z.begin(), z.end());
        double total = 0;
        for (auto& v : z) { v = std::exp(v - top); total += v; }
        for (auto& v : z) v /= total;
        return z;
    }
    const std::vector<double>& root_gumbel() const { return gumbel_; }

    sts::search::Action sampled_action(std::mt19937_64& random) const {
        std::vector<double> weights;
        for (const auto& edge : root_.edges) weights.push_back(edge.visits);
        return root_.edges[std::discrete_distribution<std::size_t>(weights.begin(), weights.end())(random)].action;
    }

    const sts::BattleContext& oracle_state() const { return particles_.front(); }

    // Advance through the exact deterministic child. If not searched (e.g. a forced move), caller builds a new root.
    bool advance(sts::search::Action action, const sts::BattleContext& played) {
        if (!settings_.oracle || root_.in_flight) throw std::logic_error{"PV: reuse requires finished oracle search"};
        if (settings_.root_halving.enabled) throw std::logic_error{"PV: root halving does not support tree reuse"};
        auto expected = particles_.front();
        action.execute(expected);
        const auto same_rng = [](const sts::Random& a, const sts::Random& b) {
            return a.counter == b.counter && a.seed0 == b.seed0 && a.seed1 == b.seed1;
        };
        const auto same_order = [](const auto& a, const auto& b) {
            if (a.size() != b.size()) return false;
            for (std::size_t i = 0; i < a.size(); ++i)
                if (a[i].uniqueId != b[i].uniqueId || a[i].id != b[i].id || a[i].upgraded != b[i].upgraded ||
                    a[i].specialData != b[i].specialData || a[i].costForTurn != b[i].costForTurn) return false;
            return true;
        };
        if (observation_key(expected) != observation_key(played) ||
            !same_rng(expected.aiRng, played.aiRng) || !same_rng(expected.shuffleRng, played.shuffleRng) ||
            !same_rng(expected.cardRandomRng, played.cardRandomRng) || !same_rng(expected.miscRng, played.miscRng) ||
            !same_rng(expected.monsterHpRng, played.monsterHpRng) || !same_rng(expected.potionRng, played.potionRng) ||
            !same_order(expected.cards.drawPile, played.cards.drawPile) ||
            !same_order(expected.cards.discardPile, played.cards.discardPile))
            throw std::logic_error{"PV: reused root differs from played state"};
        for (auto& edge : root_.edges) if (edge.key == action_key(particles_.front(), action)) {
            if (edge.outcomes.empty()) return false;
            if (edge.outcomes.size() != 1 || !edge.outcomes.contains(observation_key(played)))
                throw std::logic_error{"PV: oracle edge has inconsistent outcome"};
            auto child = std::move(edge.outcomes.begin()->second);
            if (!child->value) return false;
            root_ = std::move(*child);
            particles_.front() = played;
            simulations_ = 0;
            telemetry_ = {};
            std::vector<double> priors;
            for (const auto& next : root_.edges) priors.push_back(next.prior);
            add_root_noise(priors, settings_, random_);
            for (std::size_t i = 0; i < priors.size(); ++i) root_.edges[i].prior = priors[i];
            return true;
        }
        throw std::logic_error{"PV: played action absent from oracle root"};
    }

    // Evaluate is a batched callable: span<const Inputs> -> vector<Prediction>.
    // Keeping the boundary generic also lets algorithm checks use known predictions, not an imported runtime.
    template<class Evaluate>
    void run(Evaluate&& evaluate, std::int64_t additional_simulations) {
        if (additional_simulations < 1 || additional_simulations > std::numeric_limits<std::int64_t>::max() - simulations_ ||
            root_.in_flight) {
            throw std::invalid_argument{"PV: invalid simulation budget or unfinished batch"};
        }
        const auto budget = simulations_ + additional_simulations;
        const bool halving = settings_.root_halving.enabled;
        if (halving) {
            if (simulations_ != 0 || root_.visits != 0) throw std::logic_error{"PV: root halving runs one fresh search"};
            schedule_ = halving_schedule(std::min<int>(settings_.root_halving.max_considered, int(root_.edges.size())),
                                         additional_simulations);
        }

        while (simulations_ < budget) {
            std::vector<Request> requests;
            int pending_paths = 0;

            // Reserve paths using virtual loss. Multiple paths to one pending node share one evaluation.
            for (int slot = 0; slot < settings_.batch_size && simulations_ + pending_paths < budget; ++slot) {
                if (halving && root_.in_flight > 0 && eliminating_choice()) {
                    // A root choice that narrows the survivors waits until every in-flight path is backed up.
                    ++telemetry_.phase_flushes;
                    break;
                }
                auto state = particles_[std::uniform_int_distribution<std::size_t>{0, particles_.size() - 1}(random_)];
                Node* node = &root_;
                Path path;

                for (int depth = 0; ; ++depth) {
                    const auto moves = ordered_actions(*node, state);

                    if (!node->value) {
                        auto request = std::find_if(requests.begin(), requests.end(),
                            [node](const Request& r) { return r.node == node; });
                        if (request == requests.end()) {
                            requests.push_back({node, std::move(state), {std::move(path)}});
                        } else {
                            request->paths.push_back(std::move(path));
                        }
                        ++pending_paths;
                        break;  // never traverse an unevaluated node with guessed/uniform priors
                    }

                    if (depth == settings_.maximum_actions) {
                        backup(path, *node->value);  // explicit depth cutoff, using this node's evaluated V
                        break;
                    }

                    const auto choice = halving && node == &root_ ? root_choice() : choose(*node);
                    auto& edge = node->edges.at(choice);
                    const auto action = moves.at(choice);
                    ++node->in_flight;
                    ++edge.in_flight;
                    path.emplace_back(node, choice);

                    action.execute(state);
                    if (state.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE) {
                        throw std::runtime_error{"PV: unsupported simulator effect"};
                    }
                    if (state.outcome != sts::Outcome::UNDECIDED) {
                        backup(path, CombatObjective::terminal_value(state));
                        break;
                    }

                    auto& child = edge.outcomes[observation_key(state)];
                    if (!child) { child = std::make_unique<Node>(make_node(state)); ++telemetry_.nodes; }
                    node = child.get();
                }
            }

            // One network call for the distinct leaves, followed by value backups for all reserved paths.
            if (!requests.empty()) {
                std::vector<Inputs> batch;
                for (const auto& request : requests) {
                    batch.push_back(encode(request.state, ordered_actions(*request.node, request.state)));
                }

                const auto predictions = evaluate(std::span<const Inputs>{batch});
                ++telemetry_.batches; telemetry_.evaluated_leaves += std::int64_t(batch.size());
                if (predictions.size() != requests.size()) throw std::runtime_error{"PV: prediction count mismatch"};
                for (std::size_t i = 0; i < requests.size(); ++i) {
                    evaluate_node(*requests[i].node, predictions[i]);
                    // One rollout per evaluated node, on the reserved path's particle; its paths share the mixed value.
                    double value = predictions[i].value;
                    if (rollout_) {
                        value = (1 - settings_.rollout_mix) * value +
                                settings_.rollout_mix * (*rollout_)(requests[i].state, rollout_random_());
                    }
                    for (auto& path : requests[i].paths) backup(path, value);
                }
            }
        }
    }

private:
    using Path = std::vector<std::pair<Node*, std::size_t>>;
    struct Request { Node* node; sts::BattleContext state; std::vector<Path> paths; };

    static Node make_root(const sts::BattleContext& state) {
        validate_root_state(state);
        return make_node(state);
    }

    static Node make_node(const sts::BattleContext& state) {
        Node node;
        for (const auto action : legal_actions(state)) {
            node.edges.push_back(Edge{.action = action, .key = action_key(state, action), .outcomes = {}});
        }
        return node;
    }

    // Action indices can differ between particles (e.g. Secret Technique's draw-pile menu).
    // Match the entire legal menu by public semantic key, and return actions in the node's prior order.
    static std::vector<sts::search::Action> ordered_actions(const Node& node, const sts::BattleContext& state) {
        const auto available = legal_actions(state);
        if (available.size() != node.edges.size()) {
            throw std::runtime_error{"PV: incompatible legal menus at a shared public history"};
        }
        std::unordered_map<std::uint64_t, sts::search::Action> by_key;
        for (const auto action : available) by_key.emplace(action_key(state, action), action);

        std::vector<sts::search::Action> ordered;
        for (const auto& edge : node.edges) {
            const auto found = by_key.find(edge.key);
            if (found == by_key.end()) throw std::runtime_error{"PV: incompatible action at a shared public history"};
            ordered.push_back(found->second);
        }
        return ordered;
    }

    static void evaluate_node(Node& node, const Prediction& prediction) {
        if (node.value) throw std::logic_error{"PV: node evaluated twice"};
        const auto priors = policy_priors(prediction, node.edges.size());
        for (std::size_t i = 0; i < priors.size(); ++i) node.edges[i].prior = priors[i];
        node.value = prediction.value;
    }

    std::size_t choose(const Node& node) {
        if (!node.value) throw std::logic_error{"PV: cannot traverse an unevaluated node"};
        // Q is min-max normalized over the values backed up in this tree (MuZero), so the exploration constant is
        // independent of HP-point scales. An unvisited edge takes its parent's network value (parent FPU).
        const double span = value_hi_ > value_lo_ ? value_hi_ - value_lo_ : 0;
        std::vector<double> scores;
        for (const auto& edge : node.edges) {
            const auto visits = edge.visits + edge.in_flight;
            const double q = visits ? edge.value_sum / visits : *node.value;
            const double normalized = span > 0 ? (q - value_lo_) / span : 0.5;
            const double score = score_(normalized, edge.prior, node.visits + node.in_flight, visits);
            if (!std::isfinite(score)) throw std::runtime_error{"PV: non-finite selection score"};
            scores.push_back(score);
        }
        const auto chosen = select_(std::span<const double>{scores}, random_);
        if (chosen >= node.edges.size()) throw std::logic_error{"PV: selection policy returned an invalid edge"};
        return chosen;
    }

    void init_root_halving(const sts::BattleContext& observed, const Prediction& root_prediction) {
        const auto& h = settings_.root_halving;
        if (settings_.oracle) throw std::invalid_argument{"PV: root halving is for public-particle search only"};
        if (settings_.noise_fraction != 0) throw std::invalid_argument{"PV: root halving is evaluation-only (no root noise)"};
        if (root_prediction.logits.size() != root_.edges.size()) throw std::invalid_argument{"PV: root logits/menu mismatch"};
        if (h.max_considered < 1 || !(h.gumbel_scale >= 0) || !std::isfinite(h.gumbel_scale) || !(h.value_scale > 0) ||
            !std::isfinite(h.value_scale) || !(h.maxvisit_init >= 0) || !std::isfinite(h.maxvisit_init))
            throw std::invalid_argument{"PV: invalid root halving settings"};
        // The network's finite root logits (validated by policy_priors), not log(prior): no underflow to log 0.
        for (const float logit : root_prediction.logits) root_logits_.push_back(double(logit));
        const double top = *std::max_element(root_logits_.begin(), root_logits_.end());
        for (auto& logit : root_logits_) logit -= top;  // score_considered subtracts the max logit
        root_probs_.assign(root_logits_.size(), 0.0);  // softmax(logits): mixed-value weights only
        double total = 0;
        for (std::size_t i = 0; i < root_logits_.size(); ++i) total += root_probs_[i] = std::exp(root_logits_[i]);
        for (auto& p : root_probs_) p /= total;
        gumbel_.assign(root_.edges.size(), 0.0);
        if (h.gumbel_scale > 0) {
            // Public-observation seed only: reproducible, and independent of hidden particle contents and the tree's stream.
            std::mt19937_64 gumbel_random{observation_key(observed) ^ 0x6a09e667f3bcc909ULL};
            for (auto& g : gumbel_) {
                const double u = (double(gumbel_random() >> 11) + 0.5) * 0x1.0p-53;  // strictly inside (0, 1)
                g = h.gumbel_scale * -std::log(-std::log(u));
            }
        }
    }

    // True if the next root choice narrows the survivors: more actions sit at the scheduled considered-visit count
    // (in-flight visits included) than that value has remaining consecutive schedule entries.
    bool eliminating_choice() const {
        const auto index = std::size_t(root_.visits + root_.in_flight);
        const auto considered = schedule_.considered_visit.at(index);
        std::size_t remaining = 0;
        for (auto i = index; i < schedule_.considered_visit.size() && schedule_.considered_visit[i] == considered; ++i) ++remaining;
        std::size_t at = 0;
        for (const auto& e : root_.edges) at += e.visits + e.in_flight == considered;
        return at > remaining;
    }

    // qtransform_completed_by_mix_value at the root, from completed backups only (in-flight paths carry no value).
    std::vector<double> completed_sigma() const {
        std::vector<std::int64_t> visits; std::vector<double> sums, priors;
        for (const auto& e : root_.edges) { visits.push_back(e.visits); sums.push_back(e.value_sum); }
        return halving_completed_sigma(visits, sums, root_probs_, *root_.value, settings_.root_halving);
    }

    // score_considered argmax over actions whose visit count equals `considered` (in-flight visits included when
    // selecting a traversal). Ties: the first action in prior order, as jnp.argmax.
    std::size_t halving_argmax(std::int64_t considered, bool with_in_flight) const {
        const auto sigma = completed_sigma();
        std::size_t best = root_.edges.size();
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < root_.edges.size(); ++i) {
            const auto& e = root_.edges[i];
            if (e.visits + (with_in_flight ? e.in_flight : 0) != considered) continue;
            const double score = std::max(-1e9, gumbel_[i] + root_logits_[i] + sigma[i]);
            if (!std::isfinite(score)) throw std::runtime_error{"PV: non-finite halving score"};
            if (best == root_.edges.size() || score > best_score) { best = i; best_score = score; }
        }
        if (best == root_.edges.size()) throw std::logic_error{"PV: no root action at the considered visit count"};
        return best;
    }

    std::size_t root_choice() {
        const auto index = std::size_t(root_.visits + root_.in_flight);
        return halving_argmax(schedule_.considered_visit.at(index), true);
    }

    void backup(Path& path, double value) {
        validate_value(value);
        value_lo_ = std::min(value_lo_, value); value_hi_ = std::max(value_hi_, value);
        int turns = 0;
        for (auto [node, index] : path) {
            turns += node->edges.at(index).action.getActionType() == sts::search::ActionType::END_TURN;
            auto& edge = node->edges.at(index);
            if (node->in_flight < 1 || edge.in_flight < 1) throw std::logic_error{"PV: unreserved backup path"};
            --node->in_flight;
            --edge.in_flight;
            ++node->visits;
            ++edge.visits;
            edge.value_sum += value;
        }
        ++simulations_;
        const int depth = int(path.size());
        telemetry_.depth_sum += depth; telemetry_.turns_sum += turns;
        telemetry_.depth_max = std::max(telemetry_.depth_max, depth);
        telemetry_.turns_max = std::max(telemetry_.turns_max, turns);
        if (telemetry_.turns_hist.size() <= std::size_t(turns)) telemetry_.turns_hist.resize(turns + 1);
        ++telemetry_.turns_hist[turns];
    }

    std::vector<sts::BattleContext> particles_;
    SearchSettings settings_;
    Score score_;
    Selection select_;
    std::mt19937_64 random_;
    std::optional<GuidedRollout> rollout_;
    std::mt19937_64 rollout_random_{0x9e3779b97f4a7c15ULL};  // rollout seeds only: the tree's random_ stream is untouched
    Node root_;
    std::int64_t simulations_ = 0;
    double value_lo_ = std::numeric_limits<double>::infinity(), value_hi_ = -std::numeric_limits<double>::infinity();
    Telemetry telemetry_;
    std::vector<double> gumbel_, root_logits_, root_probs_;  // root halving only
    HalvingSchedule schedule_;
};

}  // namespace stsrl::pv
