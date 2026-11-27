#include "agents/combat/search/horizon/horizon.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace stsrl::search {
namespace {
constexpr int max_lane_steps = 64;
constexpr int max_horizon_rollout_steps = 48;

// Heavy-tailed Laplace Expected Improvement: E[max(0, s*L - g)] = 0.5 * s * exp(-g / s) for g >= 0, s > 0.
double ei_laplace(double gap, double scale) {
    if (scale <= 0.0) return 0.0;
    const double g = std::max(0.0, gap);
    return 0.5 * scale * std::exp(-g / scale);
}

std::vector<double> softmax(std::vector<double> x) {
    if (x.empty()) return {};
    const double top = *std::ranges::max_element(x);
    double sum = 0.0;
    for (auto& v : x) sum += v = std::exp(v - top);
    for (auto& v : x) v /= sum;
    return x;
}

class Particles {
public:
    Particles(const sts::BattleContext& observed, int initial, std::uint64_t stream)
        : observed_{observed}, stream_{stream}, worlds_{combat::particles(observed, std::max(1, initial), stream)} {}

    const sts::BattleContext& operator[](int world) {
        if (world >= int(worlds_.size()))
            worlds_ = combat::particles(observed_, std::max(world + 1, 2 * int(worlds_.size())), stream_);
        return worlds_[world];
    }

private:
    sts::BattleContext observed_;
    std::uint64_t stream_;
    std::vector<sts::BattleContext> worlds_;
};

void play_key(sts::BattleContext& state, std::uint64_t key) {
    const auto legal = combat::legal_actions(state);
    if (legal.empty()) return;
    for (const auto a : legal) {
        if (combat::action_key(state, a) == key) {
            a.execute(state);
            return;
        }
    }
    legal.front().execute(state);
}

void play_forced(sts::BattleContext& state, int& steps, int cap) {
    while (state.outcome == sts::Outcome::UNDECIDED && steps < cap) {
        const auto legal = combat::legal_actions(state);
        if (legal.size() != 1) return;
        legal.front().execute(state);
        ++steps;
    }
}

struct Node {
    struct Outcome {
        Node* node;
        int visits;
        bool bootstrap;  // closes a DAG cycle, so reads raw node->value instead of backed_value
    };

    bool evaluated = false;
    bool terminal = false;
    int turn = 0;
    std::uint64_t obs = 0;
    double value = 0.0;
    double backed_value = 0.0;
    // Per-world multi-turn blend value on Turn+1 boundary leaves so hidden future draws never leak across worlds.
    std::unordered_map<int, double> world_horizon_val;
    std::vector<Node*> parents;
    std::vector<std::uint64_t> keys;
    std::vector<double> logits, prior;
    int greedy_edge = 0;
    std::vector<std::vector<Outcome>> reached;
    std::vector<int> waiting;
};

int edge_visits(const Node& n, std::size_t edge) {
    int count = 0;
    for (const auto& o : n.reached[edge]) count += o.visits;
    return count;
}

int total_visits(const Node& n) {
    int count = 0;
    for (std::size_t e = 0; e < n.keys.size(); ++e) count += edge_visits(n, e);
    return count;
}

double link_value(const Node::Outcome& o) {
    return o.bootstrap ? o.node->value : o.node->backed_value;
}

double node_value_for_world(const Node& n, int world, bool bootstrap) {
    if (bootstrap) return n.value;
    if (const auto it = n.world_horizon_val.find(world); it != n.world_horizon_val.end()) return it->second;
    return n.backed_value;
}

void record_world_horizon(Node& anchor, int world, double val) {
    anchor.world_horizon_val[world] = val;
    double sum = 0.0;
    for (const auto& [w, v] : anchor.world_horizon_val) sum += v;
    anchor.backed_value = sum / double(anchor.world_horizon_val.size());
}

double edge_q(const Node& n, std::size_t edge) {
    double sum = 0.0;
    int count = 0;
    for (const auto& o : n.reached[edge]) {
        sum += o.visits * link_value(o);
        count += o.visits;
    }
    return count > 0 ? sum / double(count) : n.value;
}

bool reaches(const Node& from, const Node* to) {
    std::vector<const Node*> stack{&from};
    std::unordered_set<const Node*> seen;
    while (!stack.empty()) {
        const auto* cur = stack.back();
        stack.pop_back();
        if (cur == to) return true;
        if (!seen.insert(cur).second) continue;
        for (const auto& outcomes : cur->reached)
            for (const auto& o : outcomes)
                if (!o.bootstrap) stack.push_back(o.node);
    }
    return false;
}

bool revalue(Node& n, double readout_k0) {
    if (n.terminal || n.keys.empty()) return false;
    const double top_l = *std::ranges::max_element(n.logits);
    double best_score = -std::numeric_limits<double>::infinity();
    double best_q = n.value;
    bool any = false;
    for (std::size_t e = 0; e < n.keys.size(); ++e) {
        const int v = edge_visits(n, e);
        if (v <= 0) continue;
        any = true;
        const double q = edge_q(n, e);
        const double score = q + (n.logits[e] - top_l) / (readout_k0 + double(v));
        if (score > best_score) {
            best_score = score;
            best_q = q;
        }
    }
    const double next_val = any ? best_q : n.value;
    const bool changed = std::abs(next_val - n.backed_value) > 1e-12;
    n.backed_value = next_val;
    return changed;
}

std::size_t choose_interior(const Node& n, double c_puct) {
    const int sum = total_visits(n) + std::accumulate(n.waiting.begin(), n.waiting.end(), 0);
    const double root_n = std::sqrt(double(sum + 1));
    std::size_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n.keys.size(); ++i) {
        const int v = edge_visits(n, i);
        const double q = v > 0 ? edge_q(n, i) : n.value;
        const double score = q + c_puct * n.prior[i] * root_n / double(1 + v + n.waiting[i]);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}
}  // namespace

struct Horizon::Impl {
    struct Leaf {
        Node* node;
        std::uint64_t obs;
        sts::BattleContext state;
        std::vector<Action> actions;
    };

    struct Step {
        Node* node;
        std::size_t edge;
        Node* reached;
    };

    struct Lane {
        std::size_t root_edge = 0;
        int world = 0;
        sts::BattleContext state;
        Node* current = nullptr;
        std::vector<Step> steps;
        int plunge_choices = 0;
        int steps_taken = 0;
        bool started = false;
        int plunge_until_turn = -1;  // >= 0 while plunging greedily to next turn boundary
        bool just_expanded = false;
        bool done = false;
        Node* horizon_anchor = nullptr;  // non-null while extending Turn+1 leaf greedily for multi-turn blend
        double horizon_sum = 0.0;
        int horizon_count = 0;
        int horizon_next_turn = 0;
        int horizon_steps = 0;
        std::unordered_set<std::uint64_t> seen_obs;
    };

    enum class Phase { root, search, done };

    Settings s;
    OnWave on_wave;
    Particles particles;
    int root_turn;
    std::uint64_t root_obs;
    std::vector<Action> root_actions;
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes;
    Node& root;

    std::vector<Leaf> leaves;
    std::vector<Query> queries;
    std::unordered_set<std::uint64_t> pending_obs;

    Phase phase = Phase::root;
    int total_evals = 0;
    int sims_done = 0;
    int next_world_idx = 0;

    // Distinct-world links at root + world-difficulty-adjusted action means.
    struct WorldLink {
        Node* child;
        bool bootstrap;
    };
    std::vector<std::unordered_map<int, WorldLink>> root_worlds;
    std::vector<double> adj_q;
    std::vector<std::size_t> candidate_order;
    std::vector<Lane> active_lanes;
    std::size_t chosen_edge = 0;

    Impl(const sts::BattleContext& observed, const Settings& settings, std::uint64_t /*seed*/, OnWave callback)
        : s{settings},
          on_wave{std::move(callback)},
          particles{observed, std::max(settings.particles, settings.init_worlds),
                    observed.seed ^ (std::uint64_t(observed.turn) << 32) ^ (settings.world_draw * 0x9E3779B97F4A7C15ull)},
          root_turn{observed.turn},
          root_obs{combat::observation_key(observed)},
          root_actions{combat::legal_actions(observed)},
          root{get_or_create(root_obs)},
          root_worlds(root_actions.size()),
          adj_q(root_actions.size(), 0.0) {
        queue_leaf(&root, root_obs, observed, root_actions);
        sync_queries();
    }

    Node& get_or_create(std::uint64_t obs) {
        auto& ptr = nodes[obs];
        if (!ptr) {
            ptr = std::make_unique<Node>();
            ptr->obs = obs;
        }
        return *ptr;
    }

    void queue_leaf(Node* node, std::uint64_t obs, const sts::BattleContext& state, std::vector<Action> actions) {
        if (node->evaluated || !pending_obs.insert(obs).second) return;
        leaves.push_back({node, obs, state, std::move(actions)});
    }

    void sync_queries() {
        queries.clear();
        queries.reserve(leaves.size());
        for (const auto& leaf : leaves) queries.push_back({leaf.state, leaf.actions});
    }

    double top_root_logit() const {
        return root.logits.empty() ? 0.0 : *std::ranges::max_element(root.logits);
    }

    // Additive two-way world-effect adjustment: v_{e,w} ≈ mu_e + beta_w (sum beta_w = 0).
    // Equal to the unweighted distinct-world mean when all contenders share the same worlds,
    // and removes common world-difficulty shifts when contenders have unequal world subsets.
    void update_root_q() {
        adj_q.assign(root.keys.size(), root.value);
        std::vector<std::size_t> vis;
        std::unordered_map<int, double> beta;
        for (std::size_t e = 0; e < root.keys.size(); ++e) {
            if (root_worlds[e].empty()) continue;
            vis.push_back(e);
            double sum = 0.0;
            for (const auto& [w, l] : root_worlds[e]) {
                sum += node_value_for_world(*l.child, w, l.bootstrap);
                beta.emplace(w, 0.0);
            }
            adj_q[e] = sum / double(root_worlds[e].size());
        }
        if (vis.size() < 2 || beta.size() < 2) return;
        for (int iter = 0; iter < 6; ++iter) {
            double beta_sum = 0.0;
            for (auto& [w, bw] : beta) {
                double rsum = 0.0;
                int cnt = 0;
                for (const auto e : vis) {
                    if (const auto it = root_worlds[e].find(w); it != root_worlds[e].end()) {
                        rsum += node_value_for_world(*it->second.child, w, it->second.bootstrap) - adj_q[e];
                        ++cnt;
                    }
                }
                bw = cnt > 0 ? rsum / double(cnt) : 0.0;
                beta_sum += bw;
            }
            const double beta_mean = beta_sum / double(beta.size());
            for (auto& [w, bw] : beta) bw -= beta_mean;
            for (const auto e : vis) {
                double sum = 0.0;
                for (const auto& [w, l] : root_worlds[e])
                    sum += node_value_for_world(*l.child, w, l.bootstrap) - beta[w];
                adj_q[e] = sum / double(root_worlds[e].size());
            }
        }
    }

    double root_q(std::size_t edge) const {
        return root_worlds[edge].empty() ? root.value : adj_q[edge];
    }

    double root_score(std::size_t edge) const {
        const int v = edge_visits(root, edge);
        const double lgap = root.logits[edge] - top_root_logit();
        if (v <= 0) return root.value + lgap / s.readout_k0;
        return root_q(edge) + lgap / (s.readout_k0 + double(v));
    }

    double gap_innovation_scale(std::size_t a, std::size_t b, int w_r) const {
        const auto& wa = root_worlds[a];
        const auto& wb = root_worlds[b];
        std::vector<double> diffs;
        for (const auto& [w, la] : wa) {
            if (const auto fb = wb.find(w); fb != wb.end()) {
                const double va = node_value_for_world(*la.child, w, la.bootstrap);
                const double vb = node_value_for_world(*fb->second.child, w, fb->second.bootstrap);
                diffs.push_back(va - vb);
            }
        }
        const int n = int(diffs.size());
        double ss = 0.0;
        if (n >= 2) {
            const double mean = std::accumulate(diffs.begin(), diffs.end(), 0.0) / double(n);
            for (const double d : diffs) ss += (d - mean) * (d - mean);
        }
        const double v_mid = std::clamp(0.5 * (root_q(a) + root_q(b)), 0.0, 1.0);
        const double s0 = (2.0 * s.prior_se) * std::sqrt(v_mid * (1.0 - v_mid));
        const double se = std::sqrt(ss / double((n + 1) * (n + 1)) + (s0 * s0) / double(n + 1));
        return se * std::sqrt(double(w_r) / double(n + w_r + 1));
    }

    int active_horizon_turns() const {
        if (s.horizon_turns <= 1 || sims_done == 0) return 1;
        return s.horizon_turns;
    }

    void evaluated(std::span<const Evaluation> evals) {
        if (leaves.empty() || evals.size() != leaves.size())
            throw std::invalid_argument{"horizon: one evaluation per pending query"};
        total_evals += int(evals.size());
        for (std::size_t i = 0; i < evals.size(); ++i) {
            auto& n = *leaves[i].node;
            n.evaluated = true;
            n.turn = leaves[i].state.turn;
            n.value = double(evals[i].value);
            n.backed_value = n.value;
            n.logits.assign(evals[i].logits.begin(), evals[i].logits.end());
            n.prior = softmax(n.logits);
            n.greedy_edge = n.logits.empty() ? 0 : int(std::ranges::max_element(n.logits) - n.logits.begin());
            n.keys.clear();
            n.keys.reserve(leaves[i].actions.size());
            for (const auto a : leaves[i].actions) n.keys.push_back(combat::action_key(leaves[i].state, a));
            n.reached.resize(n.keys.size());
            n.waiting.assign(n.keys.size(), 0);
        }
        leaves.clear();
        pending_obs.clear();

        advance();
        sync_queries();
    }

    void advance() {
        for (;;) {
            if (phase == Phase::root) {
                chosen_edge = std::size_t(root.greedy_edge);
                emit_checkpoint();
                if (root_actions.size() <= 1 || std::min(root.value, 1.0 - root.value) < 5e-4 || s.simulations <= 0) {
                    phase = Phase::done;
                    return;
                }
                candidate_order.resize(root_actions.size());
                std::iota(candidate_order.begin(), candidate_order.end(), 0);
                std::ranges::stable_sort(candidate_order, std::greater{}, [&](std::size_t i) { return root.logits[i]; });
                const int width = std::min<int>({s.init_contenders, int(candidate_order.size()), s.simulations});
                const int w_cnt = std::max(1, std::min(s.init_worlds, s.simulations / std::max(1, width)));
                std::vector<std::size_t> edges(candidate_order.begin(), candidate_order.begin() + width);
                launch_wave(edges, w_cnt);
                if (active_lanes.empty()) {
                    phase = Phase::done;
                    return;
                }
                continue;
            }
            if (phase == Phase::search) {
                if (step_lanes()) return;  // waiting for GPU batch
                flush_wave();
                emit_checkpoint();
                if (!schedule_next_wave()) {
                    finalize();
                    return;
                }
                if (!leaves.empty()) return;
                continue;
            }
            return;
        }
    }

    void launch_wave(std::span<const std::size_t> edges, int n_worlds) {
        phase = Phase::search;
        active_lanes.clear();
        if (edges.empty() || n_worlds <= 0) return;
        active_lanes.reserve(edges.size() * std::size_t(n_worlds));
        for (int i = 0; i < n_worlds; ++i) {
            // Atomic per world: every active contender gets the exact same CRN worlds.
            if (sims_done + int(active_lanes.size()) + int(edges.size()) > s.simulations) break;
            const int w = next_world_idx % s.particles;
            ++next_world_idx;
            for (const auto e : edges) {
                Lane lane;
                lane.root_edge = e;
                lane.world = w;
                lane.state = particles[w];
                lane.current = &root;
                lane.seen_obs.insert(root_obs);
                active_lanes.push_back(std::move(lane));
            }
        }
    }

    bool step_lanes() {
        const int target_h = active_horizon_turns();
        for (auto& lane : active_lanes) {
            while (!lane.done) {
                if (lane.horizon_anchor != nullptr) {
                    Node* anchor = lane.horizon_anchor;
                    if (lane.state.outcome != sts::Outcome::UNDECIDED) {
                        const double win = (lane.state.outcome == sts::Outcome::PLAYER_VICTORY) ? 1.0 : 0.0;
                        lane.horizon_sum += double(target_h - lane.horizon_count) * win;
                        lane.horizon_count = target_h;
                        record_world_horizon(*anchor, lane.world, lane.horizon_sum / double(lane.horizon_count));
                        lane.done = true;
                        break;
                    }
                    const auto obs = combat::observation_key(lane.state);
                    auto& cur = get_or_create(obs);
                    if (!cur.evaluated) {
                        queue_leaf(&cur, obs, lane.state, combat::legal_actions(lane.state));
                        break;
                    }
                    while (lane.horizon_count < target_h && lane.state.turn >= lane.horizon_next_turn) {
                        lane.horizon_sum += cur.value;
                        ++lane.horizon_count;
                        ++lane.horizon_next_turn;
                    }
                    if (lane.horizon_count >= target_h || lane.horizon_steps >= max_horizon_rollout_steps ||
                        !lane.seen_obs.insert(obs).second) {
                        if (lane.horizon_count < target_h) {
                            lane.horizon_sum += double(target_h - lane.horizon_count) * cur.value;
                            lane.horizon_count = target_h;
                        }
                        record_world_horizon(*anchor, lane.world, lane.horizon_sum / double(lane.horizon_count));
                        lane.done = true;
                        break;
                    }
                    play_key(lane.state, cur.keys[std::size_t(cur.greedy_edge)]);
                    ++lane.horizon_steps;
                    play_forced(lane.state, lane.horizon_steps, max_horizon_rollout_steps);
                    continue;
                }

                if (!lane.started) {
                    lane.started = true;
                    const auto e = lane.root_edge;
                    const int prev_turn = root.turn;
                    play_key(lane.state, root.keys[e]);
                    ++lane.steps_taken;
                    play_forced(lane.state, lane.steps_taken, max_lane_steps);
                    const auto obs = combat::observation_key(lane.state);
                    auto& child = get_or_create(obs);
                    lane.steps.push_back({&root, e, &child});
                    ++root.waiting[e];
                    lane.current = &child;
                    if (lane.state.outcome != sts::Outcome::UNDECIDED) {
                        child.terminal = true;
                        child.evaluated = true;
                        child.value = child.backed_value = (lane.state.outcome == sts::Outcome::PLAYER_VICTORY) ? 1.0 : 0.0;
                        lane.done = true;
                        break;
                    }
                    const bool same_turn = (lane.state.turn == prev_turn);
                    if (!child.evaluated) {
                        lane.just_expanded = true;
                        if (same_turn) {
                            lane.plunge_until_turn = prev_turn + 1;
                            lane.plunge_choices = 1;
                        }
                        queue_leaf(&child, obs, lane.state, combat::legal_actions(lane.state));
                        break;
                    }
                    if (same_turn && total_visits(child) == 0) {
                        lane.plunge_until_turn = prev_turn + 1;
                        lane.plunge_choices = 1;
                    }
                }

                Node* cur = lane.current;
                const auto obs = cur->obs;
                if (!cur->evaluated) {
                    queue_leaf(cur, obs, lane.state, combat::legal_actions(lane.state));
                    break;
                }

                if (lane.state.turn > root_turn) {
                    if (target_h <= 1 || cur->world_horizon_val.contains(lane.world)) {
                        lane.done = true;
                        break;
                    }
                    lane.horizon_anchor = cur;
                    lane.horizon_sum = cur->value;
                    lane.horizon_count = 1;
                    lane.horizon_next_turn = lane.state.turn + 1;
                    lane.horizon_steps = 0;
                    lane.seen_obs.insert(obs);
                    play_key(lane.state, cur->keys[std::size_t(cur->greedy_edge)]);
                    ++lane.horizon_steps;
                    play_forced(lane.state, lane.horizon_steps, max_horizon_rollout_steps);
                    continue;
                }

                const bool cycle = !lane.seen_obs.insert(obs).second;
                if (cycle || lane.steps_taken >= max_lane_steps) {
                    lane.done = true;
                    break;
                }
                if (lane.plunge_until_turn >= 0) {
                    if (lane.state.turn >= lane.plunge_until_turn || lane.plunge_choices >= s.max_plunge_choices) {
                        lane.done = true;
                        break;
                    }
                } else if (lane.just_expanded) {
                    lane.done = true;
                    break;
                }

                const bool in_plunge = (lane.plunge_until_turn >= 0);
                const std::size_t edge = in_plunge ? std::size_t(cur->greedy_edge) : choose_interior(*cur, s.c_puct);
                const int prev_turn = cur->turn;
                play_key(lane.state, cur->keys[edge]);
                if (in_plunge) ++lane.plunge_choices;
                ++lane.steps_taken;
                play_forced(lane.state, lane.steps_taken, max_lane_steps);

                const auto next_obs = combat::observation_key(lane.state);
                auto& next = get_or_create(next_obs);
                lane.steps.push_back({cur, edge, &next});
                ++cur->waiting[edge];
                lane.current = &next;

                if (lane.state.outcome != sts::Outcome::UNDECIDED) {
                    next.terminal = true;
                    next.evaluated = true;
                    next.value = next.backed_value = (lane.state.outcome == sts::Outcome::PLAYER_VICTORY) ? 1.0 : 0.0;
                    lane.done = true;
                    break;
                }
                const bool same_turn = (lane.state.turn == root_turn);
                if (!next.evaluated) {
                    lane.just_expanded = true;
                    if (!in_plunge && same_turn) {
                        lane.plunge_until_turn = prev_turn + 1;
                        lane.plunge_choices = 1;
                    }
                    queue_leaf(&next, next_obs, lane.state, combat::legal_actions(lane.state));
                    break;
                }
                if (!in_plunge && same_turn && total_visits(next) == 0) {
                    lane.plunge_until_turn = prev_turn + 1;
                    lane.plunge_choices = 1;
                }
            }
        }
        return !leaves.empty();
    }

    void record_lane(const Lane& lane, std::vector<Node*>& stale, std::unordered_set<Node*>& queued) {
        for (std::size_t si = 0; si < lane.steps.size(); ++si) {
            const auto& [node, edge, reached] = lane.steps[si];
            --node->waiting[edge];
            auto& outcomes = node->reached[edge];
            auto found = std::ranges::find(outcomes, reached, &Node::Outcome::node);
            bool is_boot = false;
            if (found != outcomes.end()) {
                ++found->visits;
                is_boot = found->bootstrap;
            } else if (reaches(*reached, node)) {
                outcomes.push_back({reached, 1, true});
                is_boot = true;
            } else {
                outcomes.push_back({reached, 1, false});
                if (std::ranges::find(reached->parents, node) == reached->parents.end())
                    reached->parents.push_back(node);
            }
            if (si == 0) {
                root_worlds[edge].emplace(lane.world, WorldLink{reached, is_boot});
            }
            if (queued.insert(node).second) stale.push_back(node);
        }
    }

    void flush_wave() {
        std::vector<Node*> stale;
        std::unordered_set<Node*> queued;
        stale.reserve(active_lanes.size() * 4);
        for (const auto& lane : active_lanes) record_lane(lane, stale, queued);
        while (!stale.empty()) {
            Node* n = stale.back();
            stale.pop_back();
            queued.erase(n);
            if (revalue(*n, s.readout_k0)) {
                for (Node* p : n->parents) {
                    if (queued.insert(p).second) stale.push_back(p);
                }
            }
        }
        update_root_q();
        sims_done += int(active_lanes.size());
        active_lanes.clear();
    }

    bool schedule_next_wave() {
        const int rem = s.simulations - sims_done;
        const int w_r = std::min(s.wave_worlds, rem / 2);
        if (w_r <= 0) return false;

        std::optional<std::size_t> leader_opt;
        double leader_sc = -std::numeric_limits<double>::infinity();
        for (std::size_t e = 0; e < root.keys.size(); ++e) {
            if (edge_visits(root, e) <= 0) continue;
            const double sc = root_score(e);
            if (sc > leader_sc) {
                leader_sc = sc;
                leader_opt = e;
            }
        }
        if (!leader_opt) return false;
        const std::size_t leader = *leader_opt;
        const double top_l = top_root_logit();
        const int v_l = edge_visits(root, leader);
        const double proj_l = root_q(leader) + (root.logits[leader] - top_l) / (s.readout_k0 + double(v_l + rem / 2));

        std::optional<std::size_t> best_c;
        double best_voi = -1.0;
        for (std::size_t e = 0; e < root.keys.size(); ++e) {
            if (e == leader) continue;
            const int v_e = edge_visits(root, e);
            const double proj_e = root_q(e) + (root.logits[e] - top_l) / (s.readout_k0 + double(v_e + rem / 2));
            const double gap = (v_e > 0 && proj_e > proj_l) ? 0.0 : std::max(0.0, leader_sc - root_score(e));
            const double voi = ei_laplace(gap, gap_innovation_scale(leader, e, w_r));
            if (voi > best_voi) {
                best_voi = voi;
                best_c = e;
            }
        }
        if (!best_c) return false;

        const double avg_rate = double(total_evals) / double(std::max(1, sims_done));
        const double cost_2 = std::max(1.0, 2.0 * double(w_r) * avg_rate);
        if (s.lambda > 0.0 && best_voi <= s.lambda * cost_2) return false;

        std::vector<std::size_t> wave_edges{leader, *best_c};
        launch_wave(wave_edges, w_r);
        return !active_lanes.empty();
    }

    std::size_t best_edge() const {
        std::size_t best = std::size_t(root.greedy_edge);
        double best_sc = -std::numeric_limits<double>::infinity();
        for (std::size_t e = 0; e < root.keys.size(); ++e) {
            if (edge_visits(root, e) <= 0) continue;
            const double sc = root_score(e);
            if (sc > best_sc) {
                best_sc = sc;
                best = e;
            }
        }
        return best;
    }

    void emit_checkpoint() const {
        if (!on_wave) return;
        const auto prior = std::size_t(root.greedy_edge);
        const auto selected = sims_done ? best_edge() : prior;
        const auto estimate = [&](std::size_t e) -> std::optional<double> {
            return root_worlds[e].empty() ? std::nullopt : std::optional<double>{root_q(e)};
        };
        on_wave({sims_done, root_actions[prior], root_actions[selected],
                 estimate(prior), estimate(selected), std::max(0, total_evals - 1)});
    }

    void finalize() {
        chosen_edge = best_edge();
        phase = Phase::done;
    }

    Result result() const {
        std::vector<double> z(root.keys.size());
        for (std::size_t e = 0; e < z.size(); ++e) {
            const int v = edge_visits(root, e);
            const double q = v > 0 ? root_q(e) : root.value;
            z[e] = root.logits[e] + (s.readout_k0 + double(v)) * (q - root.value);
        }
        const auto pi = softmax(std::move(z));
        const double val = std::clamp(root_q(chosen_edge), 0.0, 1.0);
        Result out{root_actions[chosen_edge], root_actions, {}, float(val)};
        out.policy.reserve(pi.size());
        for (const double p : pi) out.policy.push_back(float(p));
        return out;
    }
};

Horizon::Horizon(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed, OnWave on_wave)
    : impl_{std::make_unique<Impl>(observed, settings, seed, std::move(on_wave))} {}
Horizon::~Horizon() = default;

std::span<const Query> Horizon::pending() const { return impl_->queries; }
void Horizon::evaluated(std::span<const Evaluation> evaluations) { impl_->evaluated(evaluations); }
Result Horizon::result() const { return impl_->result(); }
}  // namespace stsrl::search
