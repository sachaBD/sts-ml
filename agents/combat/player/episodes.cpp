#include "agents/combat/player/episodes.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/probe_grader.hpp"
#include "agents/combat/player/targets.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/state_codec.hpp"
#include <chrono>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace stsrl::player {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int sample_format = nn::layout::sample::format;
constexpr int thinking_rounds = nn::layout::sample::thinking_rounds;
constexpr auto& thinking_budgets = nn::layout::sample::thinking_budgets;
static_assert(std::size(thinking_budgets) == thinking_rounds);
constexpr double probe_probability = 0.0625;
constexpr search::Gumbel::Settings probe_settings{.simulations = 256, .considered_actions = 16, .particles = 16,
                                                  .spend_even_budget = true, .wave_checkpoints = true};
constexpr GradeSettings grade_settings{.particles = 8, .horizon_turns = 1};
using Gains = std::array<float, thinking_rounds>;

std::uint64_t probe_particle_stream(const sts::BattleContext& observed) {
    return observed.seed ^ (std::uint64_t(observed.turn) << 32) ^
           (probe_settings.world_draw * 0x9E3779B97F4A7C15ull);
}

Json checkpoint(int budget, const search::Gumbel::Round& r, std::optional<float> gain) {
    return {{"budget", budget}, {"evals", r.evals}, {"simulations", r.simulations},
            {"prior", r.prior.bits}, {"selected", r.selected.bits},
            {"prior_q", r.prior_q ? Json(*r.prior_q) : Json(nullptr)},
            {"selected_q", r.selected_q ? Json(*r.selected_q) : Json(nullptr)},
            {"gain", gain ? Json(*gain) : Json(nullptr)}};
}

struct Sample {
    std::string state;
    std::vector<search::Action> actions;
    std::vector<float> policy;
    Vitals vitals;
    Gains gains;
};

void write(std::ofstream& out, const Sample& sample, std::span<const Vitals> turn_starts, const sts::BattleContext& end) {
    const auto target = targets(sample.vitals, turn_starts, end);
    const std::uint32_t header[]{combat::state_codec_version, std::uint32_t(sample.state.size()),
                                std::uint32_t(sample.actions.size())};
    out.write(reinterpret_cast<const char*>(header), sizeof header);
    out.write(reinterpret_cast<const char*>(target.data()), sizeof target);
    out.write(reinterpret_cast<const char*>(sample.gains.data()), sizeof sample.gains);
    out.write(sample.state.data(), std::streamsize(sample.state.size()));
    for (const auto action : sample.actions)
        out.write(reinterpret_cast<const char*>(&action.bits), sizeof action.bits);
    out.write(reinterpret_cast<const char*>(sample.policy.data()), std::streamsize(sample.policy.size() * sizeof(float)));
}
}  // namespace

struct Episodes::Game {
    Json record;  // its fights-file line
    Fight fight;
    std::optional<Decision> decision, probe;
    std::optional<ProbeGrader> grader;
    std::vector<infra::Tokens> grade_requests;
    std::vector<search::Gumbel::Round> rounds;
    std::vector<std::optional<search::Gumbel::Round>> resolved;
    Json traces = Json::array();
    Gains gains;
    std::array<int, thinking_rounds> coverage{};
    std::size_t probe_evals = 0, grade_evals = 0, root_probe_evals = 0, root_grade_evals = 0;
    bool probed = false;

    void evaluated(std::span<const infra::Prediction> predictions) {
        if (probe) { probe->evaluated(predictions); return; }
        if (!grader) { decision->evaluated(predictions); return; }
        if (predictions.size() != grade_requests.size())
            throw std::invalid_argument{"grade: one prediction per request"};
        root_grade_evals += predictions.size();
        std::vector<search::Evaluation> evaluations;
        evaluations.reserve(predictions.size());
        for (const auto& p : predictions) evaluations.push_back(nn::NetworkEvaluator::answer(p));
        grader->evaluated(evaluations);
    }

    void finish_probe(const ProbeGrader* finished_grader) {
        Json checkpoints = Json::array();
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            if (!resolved[i]) continue;
            const auto& r = *resolved[i];
            const auto gain = r.selected.bits == r.prior.bits
                ? std::optional<float>{0.0f}
                : (finished_grader ? finished_grader->gain(r.selected) : std::nullopt);
            if (gain) {
                gains.at(i) = *gain;
                ++coverage.at(i);
            }
            checkpoints.push_back(checkpoint(thinking_budgets[i], r, gain));
        }
        grade_evals += root_grade_evals;
        probe_evals += root_probe_evals + root_grade_evals;
        traces.push_back({{"step", fight.steps()}, {"sample", decisions.size()},
                          {"evals", root_probe_evals + root_grade_evals},
                          {"probe_evals", root_probe_evals}, {"grade_evals", root_grade_evals},
                          {"rounds", std::move(checkpoints)}});
    }

    Answers answers;  // the last decision's, for the next
    std::vector<Sample> decisions;  // full-search roots and probe-only roots
    bool full = false;  // whether the decision in progress is a full search
    std::size_t searches = 0, full_searches = 0, network_requests = 0, reused_requests = 0;
    Clock::time_point begin = Clock::now();

    explicit Game(Json r)
        : record(std::move(r)),
          fight{combat::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()} {}

    // Play on until the search needs the network (its requests) or the game is over (none).
    std::span<const infra::Tokens> advance(const Episodes& episodes) {
        for (;;) {
            if (probe) {
                if (const auto requests = probe->pending(); !requests.empty()) return requests;
                root_probe_evals = probe->sent();
                root_grade_evals = 0;
                probe.reset();
                resolved = resolve_checkpoints(rounds, thinking_budgets);
                std::vector<search::Action> candidates;
                for (const auto& slot : resolved)
                    if (slot && slot->selected.bits != slot->prior.bits) candidates.push_back(slot->selected);
                if (candidates.empty()) {
                    finish_probe(nullptr);
                } else {
                    auto grade_random = fight.decision_random(Fight::RandomStream::thinking_grade);
                    auto stream = grade_random();
                    if (stream == probe_particle_stream(fight.state())) stream = grade_random();
                    grader.emplace(fight.state(), rounds.front().prior, candidates, stream, grade_settings);
                }
            }
            if (grader) {
                grade_requests.clear();
                for (const auto& q : grader->pending()) grade_requests.push_back(episodes.network_.request(q));
                if (!grade_requests.empty()) return grade_requests;
                finish_probe(&*grader);
                grader.reset();
            }
            if (decision) {
                if (const auto requests = decision->pending(); !requests.empty()) return requests;
                auto r = decision->result();
                if (full || probed) {
                    // Cheap roots retain only probe supervision, not a cheap policy or extra value target.
                    if (!full) r.policy.assign(r.actions.size(), 0.f);
                    Sample sample{{}, std::move(r.actions), std::move(r.policy), vitals(fight.state()), gains};
                    combat::pack(fight.state(), sample.state);
                    decisions.push_back(std::move(sample));
                }
                ++searches;
                full_searches += full;
                network_requests += decision->sent();
                reused_requests += decision->reused();
                answers = std::move(*decision).answers();
                decision.reset();
                fight.play(r.action);
            }
            if (fight.over()) return {};
            auto random = fight.decision_random();
            std::mt19937_64 probe_random{fight.decision_random(Fight::RandomStream::thinking_probe)};
            probed = episodes.stage_ == Stage::self_play && std::bernoulli_distribution(probe_probability)(probe_random);
            gains.fill(std::numeric_limits<float>::quiet_NaN());
            rounds.clear();
            if (probed) {
                probe.emplace(std::make_unique<search::Gumbel>(fight.state(), probe_settings, probe_random(),
                    [this](const search::Gumbel::Round& round) { rounds.push_back(round); }), episodes.network_);
            }
            const auto budget = episodes.budget_({fight.after_chance()}, random);
            full = budget.full;
            const auto& search = full ? episodes.full_search_ : episodes.cheap_search_;
            decision.emplace(search(fight.state(), budget.simulations, random), episodes.network_, std::move(answers));
        }
    }
};

// Lives on its owning Actor, with its fight and search.
class Episodes::Episode final : public infra::Episode {
public:
    Episode(Episodes& episodes, const Json& record) : episodes_{episodes}, game_{record} {}
    std::span<const infra::Tokens> advance() override {
        return game_.advance(episodes_);
    }
    void evaluated(std::span<const infra::Prediction> predictions) override { game_.evaluated(predictions); }
    void finish() override { episodes_.finish(game_); }
    std::string label() const override { return game_.record.at("fight_id").get<std::string>(); }

private:
    Episodes& episodes_;
    Game game_;
};

Episodes::Episodes(std::vector<Json> fights, search::SearchFactory full_search, search::SearchFactory cheap_search,
                   BudgetRule budget, Stage stage, nn::NetworkEvaluator network, const std::filesystem::path& samples,
                   std::ostream& results)
    : fights_(std::move(fights)), full_search_{std::move(full_search)}, cheap_search_{std::move(cheap_search)},  // fights_{...} would nest them in one json array
      budget_{budget}, stage_{stage}, network_{network},
      samples_{samples, std::ios::binary | std::ios::app | std::ios::ate}, sample_file_{samples.filename().string()},
      results_{results} {
    if (!samples_) throw std::runtime_error{"cannot write samples " + samples.string()};
}

std::unique_ptr<infra::Episode> Episodes::next() {
    const auto i = next_++;
    return i < fights_.size() ? std::make_unique<Episode>(*this, fights_[i]) : nullptr;
}

void Episodes::finish(const Game& g) {
    const auto& state = g.fight.state();
    const double seconds = std::chrono::duration<double>(Clock::now() - g.begin).count();
    std::lock_guard lock{output_};
    const auto begin = static_cast<std::streamoff>(samples_.tellp());
    if (begin < 0) throw std::runtime_error{"failed to locate game samples"};
    for (const auto& sample : g.decisions) write(samples_, sample, g.fight.turn_starts(), state);
    samples_.flush();
    const auto end = static_cast<std::streamoff>(samples_.tellp());
    if (!samples_ || end < begin) throw std::runtime_error{"failed to write game samples"};
    const double finished = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    Json record{{"fight_id", g.record.at("fight_id")}, {"won", state.outcome == sts::Outcome::PLAYER_VICTORY},
                     {"hp", state.player.curHp}, {"decisions", g.decisions.size()}, {"searches", g.searches},
                     {"full_searches", g.full_searches}, {"network_requests", g.network_requests},
                     {"reused_requests", g.reused_requests}, {"seconds", seconds},
                     {"turns", state.turn}, {"steps", g.fight.steps()}, {"capped", state.outcome == sts::Outcome::UNDECIDED},
                     {"finished_unix", finished}, {"sample_format", sample_format},
                     {"sample_range", {{"file", sample_file_}, {"begin", begin}, {"end", end}}}};
    if (stage_ == Stage::self_play)
        record["thinking"] = {{"evals", g.probe_evals}, {"grade_evals", g.grade_evals},
            {"coverage", g.coverage}, {"probes", g.traces},
            {"settings", {{"probability", probe_probability}, {"simulations", probe_settings.simulations},
                {"particles", probe_settings.particles}, {"considered_actions", probe_settings.considered_actions},
                {"wave", probe_settings.wave}, {"world_draw", probe_settings.world_draw},
                {"gumbel_scale", probe_settings.gumbel_scale}, {"sieve_gain", probe_settings.sieve_gain},
                {"value_scale", probe_settings.value_scale}, {"maxvisit_init", probe_settings.maxvisit_init},
                {"spend_even_budget", probe_settings.spend_even_budget}, {"rollout_mix", 0},
                {"checkpoints", std::vector<int>(std::begin(thinking_budgets), std::end(thinking_budgets))},
                {"grade_particles", grade_settings.particles},
                {"grade_horizon_turns", grade_settings.horizon_turns},
                {"grade_max_steps", grade_settings.max_steps}}}};
    results_ << record.dump() << std::endl;
    if (!samples_ || !results_) throw std::runtime_error{"failed to write game samples/results"};
    ++completed_;
    decisions_ += g.searches;
}
}  // namespace stsrl::player
