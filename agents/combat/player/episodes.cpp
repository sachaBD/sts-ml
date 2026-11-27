#include "agents/combat/player/episodes.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/fight_samples.hpp"
#include "agents/combat/player/root_supervision.hpp"
#include "environments/combat/fight_start.hpp"
#include <chrono>
#include <optional>
#include <span>
#include <stdexcept>

namespace stsrl::player {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int sample_format = nn::layout::sample::format;

}  // namespace

struct Episodes::Game {
    Json record;  // its fights-file line
    Fight fight;
    std::optional<Decision> decision;
    std::optional<RootSupervision> supervision;
    Answers answers;  // the last decision's, for the next
    FightSamples samples;
    bool full = false;  // whether the decision in progress is a full search
    std::size_t searches = 0, full_searches = 0, network_requests = 0, reused_requests = 0;
    Clock::time_point begin = Clock::now();

    Game(Json r, Stage stage)
        : record(std::move(r)),
          fight{combat::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()} {
        if (stage == Stage::self_play) supervision.emplace();
    }

    void evaluated(std::span<const infra::Prediction> predictions) {
        if (supervision && supervision->active()) supervision->evaluated(predictions);
        else decision->evaluated(predictions);
    }

    // Play on until the search needs the network (its requests) or the game is over (none).
    std::span<const infra::Tokens> advance(const Episodes& episodes) {
        for (;;) {
            if (supervision) {
                if (const auto requests = supervision->advance(); !requests.empty()) return requests;
            }
            if (decision) {
                if (const auto requests = decision->pending(); !requests.empty()) return requests;
                auto r = decision->result();
                const auto labels = supervision ? supervision->labels() : std::nullopt;
                if (full || labels) {
                    // Cheap roots retain only probe supervision, not a cheap policy or extra value target.
                    if (!full) r.policy.assign(r.actions.size(), 0.f);
                    samples.record(fight.state(), std::move(r.actions), std::move(r.policy),
                                   labels.value_or(RootSupervision::missing()));
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
            if (supervision)
                supervision->observe(fight.state(), fight.decision_random(Fight::RandomStream::thinking_probe),
                                     episodes.network_, fight.steps(), samples.size());
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
    Episode(Episodes& episodes, const Json& record) : episodes_{episodes}, game_{record, episodes.stage_} {}
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
    g.samples.write(samples_, g.fight.turn_starts(), state);
    samples_.flush();
    const auto end = static_cast<std::streamoff>(samples_.tellp());
    if (!samples_ || end < begin) throw std::runtime_error{"failed to write game samples"};
    const double finished = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    Json record{{"fight_id", g.record.at("fight_id")}, {"won", state.outcome == sts::Outcome::PLAYER_VICTORY},
                     {"hp", state.player.curHp}, {"decisions", g.samples.size()}, {"searches", g.searches},
                     {"full_searches", g.full_searches}, {"network_requests", g.network_requests},
                     {"reused_requests", g.reused_requests}, {"seconds", seconds},
                     {"turns", state.turn}, {"steps", g.fight.steps()}, {"capped", state.outcome == sts::Outcome::UNDECIDED},
                     {"finished_unix", finished}, {"sample_format", sample_format},
                     {"sample_range", {{"file", sample_file_}, {"begin", begin}, {"end", end}}}};
    if (g.supervision) record["thinking"] = g.supervision->summary();
    results_ << record.dump() << std::endl;
    if (!samples_ || !results_) throw std::runtime_error{"failed to write game samples/results"};
    ++completed_;
    decisions_ += g.searches;
}
}  // namespace stsrl::player
