// Combat episodes own simulator/search state; infra owns scheduling and inference.
#include "agents/combat/nn/decision.hpp"
#include "agents/combat/nn/targets.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include "environments/combat/public.hpp"
#include "infra/workers/actors.hpp"
#include "monitoring/telemetry.hpp"
#include "environments/combat/record_v4.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <thread>

using namespace stsrl::nn;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int sample_format = 2;  // Targets after the token and action counts

struct Sample {
    stsrl::infra::Tokens tokens;
    std::vector<float> policy;
    Vitals vitals;
};

void write(std::ofstream& out, const Sample& sample, std::span<const Vitals> turn_starts, const sts::BattleContext& end) {
    static_assert(sizeof(Targets) == 6 * 4);
    const auto& t = sample.tokens;
    const auto& policy = sample.policy;
    const auto target = targets(sample.vitals, turn_starts, end);
    out.write(reinterpret_cast<const char*>(&t.count), 4);
    out.write(reinterpret_cast<const char*>(&t.actions), 4);
    out.write(reinterpret_cast<const char*>(&target), sizeof target);
    out.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.data.size() * 4));
    out.write(reinterpret_cast<const char*>(policy.data()), std::streamsize(policy.size() * 4));
}

struct Game {
    Json record;  // its fights-file line
    Fight fight;
    double rollout_mix;
    std::optional<Decision> decision;
    std::vector<Sample> decisions;  // state, improved policy and vitals per searched move
    Clock::time_point begin = Clock::now();

    Game(Json r, double mix)
        : record(std::move(r)),
          fight{stsrl::combat_v4::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()},
          rollout_mix{mix} {}

    // Transfers the next encoded query, or returns null when the game is over.
    std::optional<stsrl::infra::Tokens> advance(const stsrl::search::Gumbel::Settings& settings) {
        for (;;) {
            if (decision) {
                if (auto query = decision->next_query()) return query;
                auto r = decision->result();
                decisions.push_back({encode(fight.state(), r.actions), std::move(r.policy), vitals(fight.state())});
                decision.reset();
                fight.play(r.action);
            }
            if (fight.over()) return std::nullopt;
            auto random = fight.decision_random();
            auto search = std::make_unique<stsrl::search::Gumbel>(fight.state(), settings, random());
            decision.emplace(fight.state(), std::move(search), rollout_mix, random());
        }
    }

    void evaluated(const stsrl::infra::Prediction& prediction) { decision->evaluated(prediction); }
};

// Everything the worker threads share.
struct Shared {
    stsrl::search::Gumbel::Settings settings;
    double rollout_mix = 0;
    std::vector<Json> fights;
    std::atomic<std::size_t> next{0};
    std::mutex output;
    std::ofstream samples;
    std::string sample_file;
    std::atomic<std::size_t> completed{0}, searched_decisions{0};

    std::unique_ptr<Game> next_game() {
        const auto i = next++;
        return i < fights.size() ? std::make_unique<Game>(fights[i], rollout_mix) : nullptr;
    }

    void finish(const Game& g) {
        const auto& state = g.fight.state();
        const float won = state.outcome == sts::Outcome::PLAYER_VICTORY;
        const double seconds = std::chrono::duration<double>(Clock::now() - g.begin).count();
        std::lock_guard lock{output};
        const auto begin = static_cast<std::streamoff>(samples.tellp());
        if (begin < 0) throw std::runtime_error{"failed to locate game samples"};
        for (const auto& sample : g.decisions) write(samples, sample, g.fight.turn_starts(), state);
        samples.flush();
        const auto end = static_cast<std::streamoff>(samples.tellp());
        if (!samples || end < begin) throw std::runtime_error{"failed to write game samples"};
        const double finished = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << Json{{"fight_id", g.record.at("fight_id")}, {"won", won > 0}, {"hp", state.player.curHp},
                          {"decisions", g.decisions.size()}, {"seconds", seconds}, {"turns", state.turn},
                          {"steps", g.fight.steps()}, {"capped", state.outcome == sts::Outcome::UNDECIDED},
                          {"finished_unix", finished}, {"sample_format", sample_format},
                          {"sample_range", {{"file", sample_file}, {"begin", begin}, {"end", end}}}}.dump() << std::endl;
        if (!samples || !std::cout) throw std::runtime_error{"failed to write game samples/results"};
        ++completed;
        searched_decisions += g.decisions.size();
    }
};

// Adapter keeps the real simulator/search and output on their owning Actor.
class CombatEpisode final : public stsrl::infra::Episode {
    std::unique_ptr<Game> game_;
    Shared& shared_;
    std::optional<std::uint64_t> pending_;
    std::uint64_t next_id_ = 0;
public:
    CombatEpisode(std::unique_ptr<Game> game, Shared& shared) : game_{std::move(game)}, shared_{shared} {}
    std::optional<stsrl::infra::EvaluationRequest> next_request() override {
        if (pending_) return std::nullopt;
        auto tokens = game_->advance(shared_.settings);
        if (!tokens) return std::nullopt;
        pending_ = next_id_++;
        return stsrl::infra::EvaluationRequest{*pending_, std::move(*tokens)};
    }
    void completed(std::uint64_t id, const stsrl::infra::Prediction& prediction) override {
        if (!pending_ || id != *pending_) throw std::logic_error{"unexpected combat evaluation"};
        game_->evaluated(prediction);
        pending_.reset();
    }
    bool done() const override { return !pending_ && !game_->decision && game_->fight.over(); }
    void finish() override { shared_.finish(*game_); }
    std::string label() const override { return game_->record.at("fight_id").get<std::string>(); }
};

void import(std::ofstream& samples) {
    using Public = sts::search::PublicBeliefCombatSearch;
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        const auto record = Json::parse(line);
        const auto& fight = record.at("fight");
        std::map<int, const Json*> searched;
        for (const auto& row : record.at("search")) searched[row.at("step").get<int>()] = &row;
        auto state = stsrl::combat_v4::start_battle(fight.at("start"));
        const bool won = fight.at("won").get<bool>();
        std::vector<Sample> decisions;
        std::vector<Vitals> turn_starts;
        observe(turn_starts, state);
        int step = 0;
        for (const auto& bits : fight.at("actions")) {
            if (searched.contains(step)) {
                const auto actions = stsrl::combat::legal_actions(state);
                std::vector<float> policy(actions.size(), 0);
                float total = 0;
                for (const auto& child : searched[step]->at("children")) {
                    const auto key = Public::publicActionKey(state, sts::search::Action{child.at("action").get<std::uint32_t>()});
                    std::size_t i = 0;
                    while (i < actions.size() && Public::publicActionKey(state, actions[i]) != key) ++i;
                    if (i == actions.size()) {
                        std::string message = "import: recorded action is not legal; fight=" +
                            fight.at("fight_id").get<std::string>() + " step=" + std::to_string(step) +
                            " action=" + std::to_string(child.at("action").get<std::uint32_t>()) + " legal=";
                        for (const auto a : actions) message += std::to_string(a.bits) + ",";
                        message += " hand=" + std::to_string(state.cards.cardsInHand) +
                            " energy=" + std::to_string(state.player.energy);
                        for (int h = 0; h < state.cards.cardsInHand; ++h) {
                            const auto& c = state.cards.hand[h];
                            message += " " + std::to_string(h) + ":" + sts::getCardEnumName(c.id) +
                                ":" + std::to_string(c.costForTurn);
                        }
                        throw std::runtime_error{message};
                    }
                    policy[i] += child.at("visits").get<float>(); total += child.at("visits").get<float>();
                }
                if (actions.size() > 1 && total > 0) {
                    for (auto& p : policy) p /= total;
                    decisions.push_back({encode(state, actions), std::move(policy), vitals(state)});
                }
            }
            sts::search::Action{bits.get<std::uint32_t>()}.execute(state);
            observe(turn_starts, state);
            ++step;
        }
        if ((state.outcome == sts::Outcome::PLAYER_VICTORY) != won) throw std::runtime_error{"import: fight did not replay"};
        for (const auto& sample : decisions) write(samples, sample, turn_starts, state);
    }
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string{argv[1]} == "check") {
        // Capability/shape preflight only: no network, search, or game outcomes.
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            const auto fight = Json::parse(line);
            const auto state = stsrl::combat_v4::start_battle(fight.at("start"));
            const auto tokens = encode(state, stsrl::combat::legal_actions(state));
            std::cout << Json{{"fight_id", fight.at("fight_id")}, {"tokens", tokens.count},
                              {"actions", tokens.actions}}.dump() << std::endl;
        }
        return 0;
    }
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "import" && argc == 3) {
        std::ofstream samples{argv[2], std::ios::binary | std::ios::app};
        import(samples);
        return 0;
    }
    if (mode != "play" || argc < 4 || argc % 2) {
        std::cerr << "usage: nn_play play MODEL.onnx SAMPLES.bin [--threads T] [--parallel G] [--gpu-model FIXED.onnx\n"
                     "                    --gpu-batch B] [--gpu-bank SMALLER.onnx,...] [--gpu-wait-us U]\n"
                     "                    [--gpu-attended A] [--gpu-wide-model WIDE.onnx] [--gpu-wide-attended A] [--cpu-lanes L]\n"
                     "                    [--sims S] [--rollout-mix M] [--value-scale V]\n"
                     "                    [--gumbel-scale G] [--particles P] [--timeout-ms M] [--shutdown-timeout-ms M]\n"
                     "                    [--telemetry TELEMETRY.jsonl]\n"
                     "       nn_play import SAMPLES.bin\n";
        return 2;
    }
    int threads = 9, parallel = 224, gpu_batch = 512, gpu_wait_us = 100, cpu_lanes = 2;
    int timeout_ms = 60'000, shutdown_timeout_ms = 5'000;
    std::string gpu_model, gpu_wide_model, telemetry;
    int gpu_attended = 0, gpu_wide_attended = 0;
    std::vector<std::filesystem::path> gpu_bank;
    stsrl::search::Gumbel::Settings settings;
    double rollout_mix = 0.5;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--threads") threads = std::stoi(v);
        else if (k == "--cpu-lanes") cpu_lanes = std::stoi(v);
        else if (k == "--parallel") parallel = std::stoi(v);
        else if (k == "--timeout-ms") timeout_ms = std::stoi(v);
        else if (k == "--shutdown-timeout-ms") shutdown_timeout_ms = std::stoi(v);
        else if (k == "--telemetry") telemetry = v;
        else if (k == "--gpu-wait-us") gpu_wait_us = std::stoi(v);
        else if (k == "--gpu-bank") {
            std::size_t begin = 0;
            for (std::size_t end; begin < v.size(); begin = end + 1) {
                end = v.find(',', begin);
                gpu_bank.emplace_back(v.substr(begin, end - begin));
                if (end == std::string::npos) break;
            }
        }
        else if (k == "--gpu-model") gpu_model = v;
        else if (k == "--gpu-batch") gpu_batch = std::stoi(v);
        else if (k == "--gpu-attended") gpu_attended = std::stoi(v);
        else if (k == "--gpu-wide-model") gpu_wide_model = v;
        else if (k == "--gpu-wide-attended") gpu_wide_attended = std::stoi(v);
        else if (k == "--sims") settings.simulations = std::stoi(v);
        else if (k == "--value-scale") settings.value_scale = std::stod(v);
        else if (k == "--rollout-mix") rollout_mix = std::stod(v);
        else if (k == "--gumbel-scale") settings.gumbel_scale = std::stod(v);
        else if (k == "--particles") settings.particles = std::stoi(v);
        else throw std::invalid_argument{"unknown option " + k};
    }
    if (threads < 1 || parallel < 1 || gpu_batch < 1 || cpu_lanes < 1 || gpu_wait_us < 0 || timeout_ms <= 0 || shutdown_timeout_ms <= 0 ||
        settings.simulations < 0 || settings.particles < 1 || rollout_mix < 0 || rollout_mix > 1 || settings.gumbel_scale < 0 ||
        !std::isfinite(settings.value_scale) || settings.value_scale < 0)
        throw std::invalid_argument{"positive threads/slots/batch/particles/timeouts, nonnegative sims, rollout mix in [0,1], nonnegative noise/wait, finite nonnegative value scale required"};
    if (gpu_model.empty() && (!gpu_bank.empty() || !gpu_wide_model.empty()))
        throw std::invalid_argument{"GPU bank and wide model require gpu-model"};
    if (gpu_attended < 0 || gpu_wide_attended < 0) throw std::invalid_argument{"nonnegative attention capacities required"};
    const auto setup_begin = Clock::now();
    if (!gpu_model.empty()) gpu_bank.push_back(gpu_model);
    auto narrow = gpu_model.empty() ? stsrl::infra::cpu_backend(argv[2]) : stsrl::infra::gpu_backend(argv[2], gpu_bank, 2, gpu_attended);
    auto wide = gpu_wide_model.empty() ? nullptr : stsrl::infra::gpu_backend(argv[2], {gpu_wide_model}, 1, gpu_wide_attended);
    // Rows no GPU model accepts get their own queue, so they never hold up a GPU lane.
    auto overflow = gpu_model.empty() ? nullptr : stsrl::infra::cpu_backend(argv[2], cpu_lanes);
    std::vector<stsrl::infra::Backend*> backends{narrow.get()};
    if (wide) backends.push_back(wide.get());
    if (overflow) backends.push_back(overflow.get());
    for (const auto* backend : backends)
        if (backend->width() != width) throw std::invalid_argument{"model token width does not match combat encoder"};
    std::cerr << "timing: evaluator setup "
              << std::chrono::duration<double>(Clock::now() - setup_begin).count() << " s\n";
    Shared shared{settings, rollout_mix, {}, {}, {},
                  std::ofstream{argv[3], std::ios::binary | std::ios::app | std::ios::ate},
                  std::filesystem::path{argv[3]}.filename().string()};
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) shared.fights.push_back(Json::parse(line));
    const auto play_begin = Clock::now();
    const double play_wall_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    stsrl::infra::ActorConfig config;
    config.actors = threads; config.slots = parallel; config.batch = gpu_batch;
    config.batch_wait = std::chrono::microseconds{gpu_wait_us};
    config.timeout = std::chrono::milliseconds{timeout_ms};
    config.shutdown_timeout = std::chrono::milliseconds{shutdown_timeout_ms};
    std::optional<stsrl::monitoring::TelemetryFile> telemetry_file;
    if (!telemetry.empty()) {
        telemetry_file.emplace(telemetry);
        config.on_counters = [&](const stsrl::infra::Counters& counters) { telemetry_file->write(counters); };
    }
    try {
        stsrl::infra::run_actors(config, [&]() -> std::unique_ptr<stsrl::infra::Episode> {
            auto game = shared.next_game();
            return game ? std::make_unique<CombatEpisode>(std::move(game), shared) : nullptr;
        }, backends);
    } catch (const std::exception& e) {
        if (std::string{e.what()}.find("CUDA") != std::string::npos) {
            // A poisoned CUDA context requires a fresh process, not backend teardown.
            std::cerr << "nn_play: CUDA failure; exiting without GPU teardown: " << e.what() << '\n';
            std::fflush(nullptr);
            std::_Exit(1);
        }
        throw;
    }
    if (shared.completed != shared.fights.size()) throw std::runtime_error{"missing completed fights"};
    const double gameplay_seconds = std::chrono::duration<double>(Clock::now() - play_begin).count();
    std::cerr << "timing: gameplay " << gameplay_seconds << " s\n";
    std::cerr << "play_stats: " << Json{{"seconds", gameplay_seconds}, {"wall_start_unix", play_wall_start}, {"games", shared.completed.load()},
        {"searched_decisions", shared.searched_decisions.load()},
        {"decisions_per_second", shared.searched_decisions.load() / gameplay_seconds},
        {"games_per_hour", shared.completed.load() * 3600. / gameplay_seconds}}.dump() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "nn_play: " << e.what() << "\n";
    return 1;
}
