// Samples one uniform K>=2 decision per fight and scores it with 2-draw reference Gumbel
// and (optionally) a Horizon probe + 2-draw coupled ProbeGrader at H=1 and fight-end.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/probe_grader.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "agents/combat/search/horizon/horizon.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include "infra/workers/actors.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace stsrl;
using Json = nlohmann::json;
constexpr std::array<std::uint64_t, 2> reference_draws{1'000'001ull, 1'000'002ull};

struct Config {
    int sims = 2048;
    int particles = 64;
    int considered_actions = 128;
    bool horizon_probe = true;
    int probe_sims = 160;
    int probe_particles = 16;
    int grade_particles = 16;
};

std::uint64_t probe_particle_stream(const sts::BattleContext& observed, std::uint64_t world_draw) {
    return observed.seed ^ (std::uint64_t(observed.turn) << 32) ^ (world_draw * 0x9E3779B97F4A7C15ull);
}

std::size_t pick_decision_index(std::string_view fight_id, std::size_t count) {
    std::seed_seq seeds(fight_id.begin(), fight_id.end());
    std::mt19937_64 rng{seeds};
    return std::uniform_int_distribution<std::size_t>{0, count - 1}(rng);
}

class Output {
public:
    explicit Output(const std::filesystem::path& states_path)
        : states_{states_path, std::ios::binary | std::ios::trunc} {
        if (!states_) throw std::runtime_error{"cannot open states file: " + states_path.string()};
    }

    void write_skipped(std::size_t index, const std::string& fight_id, bool fight_won) {
        std::lock_guard lock{mutex_};
        Json row{{"index", index}, {"fight_id", fight_id}, {"fight_won", fight_won}, {"decisions", 0}};
        std::cout << row.dump() << std::endl;
        if (!std::cout) throw std::runtime_error{"failed writing stdout"};
    }

    void write(Json row, const sts::BattleContext& state, std::span<const search::Action> actions) {
        std::string blob;
        combat::pack(state, blob);
        const std::uint32_t header[]{combat::state_codec_version, std::uint32_t(blob.size()),
                                     std::uint32_t(actions.size())};
        std::lock_guard lock{mutex_};
        const auto begin = static_cast<std::streamoff>(states_.tellp());
        states_.write(reinterpret_cast<const char*>(header), sizeof header);
        states_.write(blob.data(), std::streamsize(blob.size()));
        for (const auto a : actions)
            states_.write(reinterpret_cast<const char*>(&a.bits), sizeof a.bits);
        states_.flush();
        const auto end = static_cast<std::streamoff>(states_.tellp());
        if (!states_ || end <= begin) throw std::runtime_error{"failed writing states file"};
        row["state_range"] = {{"begin", begin}, {"end", end}};
        std::cout << row.dump() << std::endl;
        if (!std::cout) throw std::runtime_error{"failed writing stdout"};
    }

private:
    std::mutex mutex_;
    std::ofstream states_;
};

struct Visited {
    int step = 0;
    int turn = 0;
    std::vector<std::uint32_t> moves;
    sts::BattleContext state;
    std::mt19937_64 play_rng;
    std::mt19937_64 probe_rng;
    std::mt19937_64 grade_rng;
};

class CollectEpisode final : public infra::Episode {
public:
    CollectEpisode(std::size_t index, const Json& record, const Config& cfg,
                   const nn::NetworkEvaluator& evaluator, Output& out)
        : index_{index},
          fight_id_{record.at("fight_id").get<std::string>()},
          fight_{combat::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()},
          cfg_{cfg},
          evaluator_{evaluator},
          out_{out} {}

    std::span<const infra::Tokens> advance() override {
        for (;;) {
            if (phase_ == Phase::walk) {
                if (decision_) {
                    if (const auto reqs = decision_->pending(); !reqs.empty()) return reqs;
                    const auto r = decision_->result();
                    answers_ = std::move(*decision_).answers();
                    decision_.reset();
                    visited_.push_back(Visited{
                        fight_.steps(),
                        fight_.state().turn,
                        moves_,
                        fight_.state(),
                        fight_.decision_random(player::Fight::RandomStream::play),
                        fight_.decision_random(player::Fight::RandomStream::thinking_probe),
                        fight_.decision_random(player::Fight::RandomStream::thinking_grade),
                    });
                    moves_.push_back(r.action.bits);
                    fight_.play(r.action);
                }
                if (!fight_.over()) {
                    search::Gumbel::Settings greedy{.simulations = 0, .gumbel_scale = 0.0, .particles = 1};
                    auto rng = fight_.decision_random();
                    decision_.emplace(std::make_unique<search::Gumbel>(fight_.state(), greedy, rng()),
                                      evaluator_, std::move(answers_));
                    continue;
                }
                fight_won_ = fight_.state().outcome == sts::Outcome::PLAYER_VICTORY;
                total_decisions_ = visited_.size();
                if (visited_.empty()) {
                    phase_ = Phase::done;
                    return {};
                }
                sampled_index_ = pick_decision_index(fight_id_, visited_.size());
                chosen_.emplace(std::move(visited_[sampled_index_]));
                visited_.clear();
                visited_.shrink_to_fit();
                actions_ = combat::legal_actions(chosen_->state);
                phase_ = Phase::reference;
                draw_idx_ = 0;
                start_reference_draw();
                continue;
            }

            if (phase_ == Phase::reference) {
                if (const auto reqs = decision_->pending(); !reqs.empty()) return reqs;
                record_reference_draw();
                ++draw_idx_;
                if (draw_idx_ < reference_draws.size()) {
                    start_reference_draw();
                    continue;
                }
                if (!cfg_.horizon_probe) {
                    phase_ = Phase::done;
                    return {};
                }
                phase_ = Phase::probe;
                search::Horizon::Settings ps{
                    .simulations = cfg_.probe_sims, .particles = cfg_.probe_particles, .lambda = 0.0};
                probe_stream_ = probe_particle_stream(chosen_->state, ps.world_draw);
                decision_.emplace(
                    std::make_unique<search::Horizon>(chosen_->state, ps, chosen_->probe_rng(),
                                                      [this](const search::Checkpoint& w) { waves_.push_back(w); }),
                    evaluator_);
                continue;
            }

            if (phase_ == Phase::probe) {
                if (const auto reqs = decision_->pending(); !reqs.empty()) return reqs;
                probe_evals_ = decision_->sent();
                decision_.reset();
                resolved_ = player::resolve_checkpoints(waves_, nn::layout::sample::thinking_budgets);
                std::vector<search::Action> candidates;
                for (const auto& slot : resolved_)
                    if (slot && slot->selected.bits != slot->prior.bits) candidates.push_back(slot->selected);
                if (candidates.empty()) {
                    phase_ = Phase::done;
                    return {};
                }
                candidates_ = std::move(candidates);
                phase_ = Phase::grade;
                grade_stage_ = 0;
                start_grader();
                continue;
            }

            if (phase_ == Phase::grade) {
                grade_requests_.clear();
                for (const auto& q : grader_->pending()) grade_requests_.push_back(evaluator_.request(q));
                if (!grade_requests_.empty()) return grade_requests_;
                finish_grader();
                ++grade_stage_;
                if (grade_stage_ < 4) {
                    start_grader();
                    continue;
                }
                phase_ = Phase::done;
                return {};
            }

            return {};
        }
    }

    void evaluated(std::span<const infra::Prediction> predictions) override {
        if (phase_ == Phase::grade) {
            if (predictions.size() != grade_requests_.size())
                throw std::invalid_argument{"grade: prediction count mismatch"};
            grade_evals_ += predictions.size();
            std::vector<search::Evaluation> evals;
            evals.reserve(predictions.size());
            for (const auto& p : predictions) evals.push_back(nn::NetworkEvaluator::answer(p));
            grader_->evaluated(evals);
            return;
        }
        decision_->evaluated(predictions);
    }

    void finish() override {
        if (!chosen_) {
            out_.write_skipped(index_, fight_id_, fight_won_);
            return;
        }
        Json row{
            {"index", index_},
            {"fight_id", fight_id_},
            {"fight_won", fight_won_},
            {"decisions", total_decisions_},
            {"sampled_decision_index", sampled_index_},
            {"step", chosen_->step},
            {"turn", chosen_->turn},
            {"moves", chosen_->moves},
            {"draws", std::move(draws_json_)},
        };
        if (cfg_.horizon_probe) {
            Json rounds = Json::array();
            const auto& budgets = nn::layout::sample::thinking_budgets;
            for (std::size_t i = 0; i < resolved_.size(); ++i) {
                if (!resolved_[i]) continue;
                const auto& r = *resolved_[i];
                const bool same = r.selected.bits == r.prior.bits;
                rounds.push_back({
                    {"budget", budgets[i]},
                    {"evals", r.evals},
                    {"simulations", r.simulations},
                    {"prior", r.prior.bits},
                    {"selected", r.selected.bits},
                    {"gain_h1_a", same ? 0.0f : gains_[0][i]},
                    {"gain_h1_b", same ? 0.0f : gains_[1][i]},
                    {"gain_end_a", same ? 0.0f : gains_[2][i]},
                    {"gain_end_b", same ? 0.0f : gains_[3][i]},
                });
            }
            row["horizon_probe"] = {
                {"probe_evals", probe_evals_},
                {"grade_evals", grade_evals_},
                {"rounds", std::move(rounds)},
            };
        }
        out_.write(std::move(row), chosen_->state, actions_);
    }

    std::string label() const override { return fight_id_; }

private:
    enum class Phase { walk, reference, probe, grade, done };

    void start_reference_draw() {
        search::Gumbel::Settings s{
            .simulations = cfg_.sims,
            .considered_actions = cfg_.considered_actions,
            .gumbel_scale = 0.0,
            .particles = cfg_.particles,
            .world_draw = reference_draws[draw_idx_],
        };
        auto gumbel = std::make_unique<search::Gumbel>(chosen_->state, s, chosen_->play_rng());
        gumbel_ptr_ = gumbel.get();
        decision_.emplace(std::move(gumbel), evaluator_);
    }

    void record_reference_draw() {
        const auto r = decision_->result();
        const auto graph = gumbel_ptr_->graph();
        const auto root_it = std::ranges::find(graph.nodes, graph.root, &search::Gumbel::Graph::Node::observation);
        if (root_it == graph.nodes.end() || root_it->edges.size() != r.actions.size())
            throw std::logic_error{"thinking_eval_collect: invalid root graph"};
        Json edges = Json::array();
        for (std::size_t i = 0; i < r.actions.size(); ++i) {
            const auto& e = root_it->edges[i];
            edges.push_back({
                {"bits", r.actions[i].bits},
                {"logit", e.logit},
                {"prior", e.prior},
                {"visits", e.visits},
                {"q", e.q ? Json(*e.q) : Json(nullptr)},
                {"improved_policy", e.improved_policy},
            });
        }
        draws_json_.push_back({
            {"world_draw", reference_draws[draw_idx_]},
            {"action", r.action.bits},
            {"v_net", root_it->evaluation_value},
            {"backed_value", root_it->backed_value},
            {"network_requests", decision_->sent()},
            {"edges", std::move(edges)},
        });
        gumbel_ptr_ = nullptr;
        decision_.reset();
    }

    std::uint64_t fresh_grade_stream() {
        for (;;) {
            const auto s = chosen_->grade_rng();
            if (s != probe_stream_ && std::ranges::find(used_streams_, s) == used_streams_.end()) {
                used_streams_.push_back(s);
                return s;
            }
        }
    }

    void start_grader() {
        const bool full_fight = grade_stage_ >= 2;
        const player::GradeSettings gs{
            .particles = cfg_.grade_particles,
            .horizon_turns = full_fight ? 50 : 1,
            .max_steps = full_fight ? 512 : 64,
        };
        grader_.emplace(chosen_->state, waves_.front().prior, candidates_, fresh_grade_stream(), gs);
    }

    void finish_grader() {
        for (std::size_t i = 0; i < resolved_.size(); ++i) {
            if (!resolved_[i]) continue;
            const auto g = grader_->gain(resolved_[i]->selected);
            gains_[grade_stage_][i] = g.value_or(0.0f);
        }
        grader_.reset();
    }

    std::size_t index_;
    std::string fight_id_;
    player::Fight fight_;
    Config cfg_;
    const nn::NetworkEvaluator& evaluator_;
    Output& out_;

    Phase phase_ = Phase::walk;
    player::Answers answers_;
    std::optional<player::Decision> decision_;
    const search::Gumbel* gumbel_ptr_ = nullptr;
    std::vector<std::uint32_t> moves_;
    std::vector<Visited> visited_;
    bool fight_won_ = false;
    std::size_t total_decisions_ = 0;
    std::size_t sampled_index_ = 0;
    std::optional<Visited> chosen_;
    std::vector<search::Action> actions_;

    std::size_t draw_idx_ = 0;
    Json draws_json_ = Json::array();

    std::uint64_t probe_stream_ = 0;
    std::vector<std::uint64_t> used_streams_;
    std::vector<search::Checkpoint> waves_;
    std::vector<std::optional<search::Checkpoint>> resolved_;
    std::vector<search::Action> candidates_;
    std::optional<player::ProbeGrader> grader_;
    std::vector<infra::Tokens> grade_requests_;
    std::size_t probe_evals_ = 0;
    std::size_t grade_evals_ = 0;
    std::size_t grade_stage_ = 0;
    std::array<std::array<float, nn::layout::sample::thinking_rounds>, 4> gains_{};
};
}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"Walks fights on stdin, uniformly samples one K>=2 state per fight, and runs 2-draw reference Gumbel "
                 "plus optional Horizon+ProbeGrader checks.", "thinking_eval_collect"};
    std::filesystem::path models, states_path;
    Config cfg;
    bool cpu = false;
    int actors = 9, slots = 224;
    app.add_option("MODELS", models, "exported model directory")->required();
    app.add_option("STATES_BIN", states_path, "output binary state blob file")->required();
    app.add_option("--sims", cfg.sims, "reference Gumbel simulations per draw");
    app.add_option("--particles", cfg.particles, "reference Gumbel particles per draw");
    app.add_option("--considered-actions", cfg.considered_actions, "reference Gumbel considered actions");
    app.add_flag("--horizon-probe,!--no-horizon-probe", cfg.horizon_probe, "also run Horizon probe + 2-draw ProbeGrader");
    app.add_option("--probe-sims", cfg.probe_sims, "Horizon probe simulations");
    app.add_option("--probe-particles", cfg.probe_particles, "Horizon probe particles");
    app.add_option("--grade-particles", cfg.grade_particles, "ProbeGrader particles per draw");
    app.add_option("--actors", actors, "worker actor threads");
    app.add_option("--slots", slots, "episodes in flight per actor");
    app.add_flag("--cpu", cpu, "evaluate on CPU only");
    CLI11_PARSE(app, argc, argv);

    std::vector<Json> fights;
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) fights.push_back(Json::parse(line));

    const auto network = nn::load_network(models, cpu ? nn::Backends::cpu : nn::Backends::all);
    const nn::NetworkEvaluator evaluator{network.encoding};
    std::vector<infra::Backend*> backends;
    for (const auto& b : network.backends) backends.push_back(b.get());

    Output out{states_path};
    std::atomic<std::size_t> next{0};
    infra::ActorConfig worker_cfg{};
    worker_cfg.actors = actors;
    worker_cfg.slots = slots;
    infra::run_actors(worker_cfg, [&]() -> std::unique_ptr<infra::Episode> {
        const auto i = next++;
        return i < fights.size() ? std::make_unique<CollectEpisode>(i, fights[i], cfg, evaluator, out) : nullptr;
    }, backends);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "thinking_eval_collect: " << e.what() << '\n';
    return 1;
}
