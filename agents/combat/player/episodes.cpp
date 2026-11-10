#include "agents/combat/player/episodes.hpp"
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/targets.hpp"
#include "environments/combat/fight_start.hpp"
#include <chrono>
#include <optional>
#include <span>
#include <stdexcept>

namespace stsrl::player {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
// A sample record: token count, action count, token width, targets, tokens, policy.
constexpr int sample_format = nn::layout::sample::format;

struct Sample {
    infra::Tokens tokens;
    std::vector<float> policy;
    Vitals vitals;
};

void write(std::ofstream& out, const Sample& sample, std::span<const Vitals> turn_starts, const sts::BattleContext& end) {
    const auto& t = sample.tokens;
    const auto target = targets(sample.vitals, turn_starts, end);
    const auto width = std::int32_t(t.data.size() / t.count);
    out.write(reinterpret_cast<const char*>(&t.count), 4);
    out.write(reinterpret_cast<const char*>(&t.actions), 4);
    out.write(reinterpret_cast<const char*>(&width), 4);
    out.write(reinterpret_cast<const char*>(&target), sizeof target);
    out.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.data.size() * 4));
    out.write(reinterpret_cast<const char*>(sample.policy.data()), std::streamsize(sample.policy.size() * 4));
}
}  // namespace

struct Episodes::Game {
    Json record;  // its fights-file line
    Fight fight;
    std::optional<Decision> decision;
    std::vector<Sample> decisions;  // state, improved policy and vitals per searched move
    Clock::time_point begin = Clock::now();

    explicit Game(Json r)
        : record(std::move(r)),
          fight{combat::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()} {}

    // Play on until the search needs the network (its query) or the game is over (nullptr).
    const infra::Tokens* advance(const search::SearchFactory& search, const nn::NetworkEvaluator& network) {
        for (;;) {
            if (decision) {
                if (const auto query = decision->pending()) return query;
                auto r = decision->result();
                decisions.push_back({network.request({fight.state(), r.actions}), std::move(r.policy), vitals(fight.state())});
                decision.reset();
                fight.play(r.action);
            }
            if (fight.over()) return nullptr;
            auto random = fight.decision_random();
            decision.emplace(search(fight.state(), random), network);
        }
    }
};

// Lives on its owning Actor, with its fight and search.
class Episodes::Episode final : public infra::Episode {
public:
    Episode(Episodes& episodes, const Json& record) : episodes_{episodes}, game_{record} {}
    const infra::Tokens* advance() override { return game_.advance(episodes_.search_, episodes_.network_); }
    void evaluated(const infra::Prediction& prediction) override { game_.decision->evaluated(prediction); }
    void finish() override { episodes_.finish(game_); }
    std::string label() const override { return game_.record.at("fight_id").get<std::string>(); }

private:
    Episodes& episodes_;
    Game game_;
};

Episodes::Episodes(std::vector<Json> fights, search::SearchFactory search, nn::NetworkEvaluator network,
                   const std::filesystem::path& samples, std::ostream& results)
    : fights_(std::move(fights)), search_{std::move(search)}, network_{network},  // fights_{...} would nest them in one json array
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
    results_ << Json{{"fight_id", g.record.at("fight_id")}, {"won", state.outcome == sts::Outcome::PLAYER_VICTORY},
                     {"hp", state.player.curHp}, {"decisions", g.decisions.size()}, {"seconds", seconds},
                     {"turns", state.turn}, {"steps", g.fight.steps()}, {"capped", state.outcome == sts::Outcome::UNDECIDED},
                     {"finished_unix", finished}, {"sample_format", sample_format},
                     {"sample_range", {{"file", sample_file_}, {"begin", begin}, {"end", end}}}}.dump() << std::endl;
    if (!samples_ || !results_) throw std::runtime_error{"failed to write game samples/results"};
    ++completed_;
    decisions_ += g.decisions.size();
}
}  // namespace stsrl::player
