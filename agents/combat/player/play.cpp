// Combat episodes own simulator/search state; infra owns scheduling and inference.
#include "agents/combat/player/decision.hpp"
#include "agents/combat/player/targets.hpp"
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/config.hpp"
#include "environments/combat/public.hpp"
#include "infra/workers/actors.hpp"
#include "infra/workers/telemetry.hpp"
#include "environments/combat/fight_start.hpp"
#include <atomic>
#include <chrono>
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

using namespace stsrl::player;
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
    std::optional<Decision> decision;
    std::vector<Sample> decisions;  // state, improved policy and vitals per searched move
    Clock::time_point begin = Clock::now();

    explicit Game(Json r)
        : record(std::move(r)),
          fight{stsrl::combat::start_battle(record.at("start")), record.at("start").at("seed").get<std::uint64_t>()} {}

    // Play on until the search needs the network (its query) or the game is over (nullptr).
    const stsrl::infra::Tokens* advance(const stsrl::search::SearchFactory& search, const NetworkEvaluator& network) {
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

    void evaluated(const stsrl::infra::Prediction& prediction) { decision->evaluated(prediction); }
};

// Everything the worker threads share.
struct Shared {
    stsrl::search::SearchFactory search;
    NetworkEvaluator network;
    std::vector<Json> fights;
    std::atomic<std::size_t> next{0};
    std::mutex output;
    std::ofstream samples;
    std::string sample_file;
    std::atomic<std::size_t> completed{0}, searched_decisions{0};

    std::unique_ptr<Game> next_game() {
        const auto i = next++;
        return i < fights.size() ? std::make_unique<Game>(fights[i]) : nullptr;
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
public:
    CombatEpisode(std::unique_ptr<Game> game, Shared& shared) : game_{std::move(game)}, shared_{shared} {}
    const stsrl::infra::Tokens* advance() override {
        return game_->advance(shared_.search, shared_.network);
    }
    void evaluated(const stsrl::infra::Prediction& prediction) override { game_->evaluated(prediction); }
    void finish() override { shared_.finish(*game_); }
    std::string label() const override { return game_->record.at("fight_id").get<std::string>(); }
};

struct Network {
    Encoding encoding;
    std::vector<std::unique_ptr<stsrl::infra::Backend>> backends;  // in the order requests try them
};

// network.json is written by agents/combat/nn/pytorch_to_onnx.py.
Network load_network(const std::filesystem::path& models) {
    std::ifstream file{models / "network.json"};
    if (!file) throw std::invalid_argument{"no network.json in " + models.string()};
    const auto manifest = Json::parse(file);
    Network network{encoding_from_name(manifest.at("encoding").get<std::string>()), {}};
    for (const auto& deployment : manifest.at("deployments")) {
        std::vector<std::filesystem::path> files;
        for (const auto& name : deployment.at("files")) files.push_back(models / name.get<std::string>());
        network.backends.push_back(stsrl::infra::gpu_backend(files, deployment.at("lanes").get<int>()));
    }
    if (network.backends.empty()) throw std::invalid_argument{"network.json has no deployments"};
    const auto& cpu = manifest.at("cpu");
    network.backends.push_back(stsrl::infra::cpu_backend(models / cpu.at("file").get<std::string>(),
                                                         cpu.at("lanes").get<int>(), cpu.at("batch").get<int>()));
    for (const auto& backend : network.backends)
        if (backend->width() != width) throw std::invalid_argument{backend->name() + " model token width does not match combat encoder"};
    return network;
}
}  // namespace

int main(int argc, char** argv) try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode != "play" || argc < 4 || argc % 2) {
        std::cerr << "usage: nn_play play MODELS SAMPLES.bin OPTIONS < STARTS.jsonl > RESULTS.jsonl\n"
                     "Plays each fight start on stdin; appends its samples to SAMPLES.bin and prints one result line.\n"
                     "MODELS is an exported model directory: its network.json names the encoding, the GPU deployments\n"
                     "(smallest token capacity first) and the CPU model, with their lanes. Each input goes to the smallest\n"
                     "deployment that fits it, else the CPU.\n"
                     "Every option is required:\n"
                     "  --threads T                actors (CPU threads playing fights)\n"
                     "  --parallel G               fights in flight per actor\n"
                     "  --gpu-wait-us U            longest a partial inference batch waits for more requests\n"
                     "  --timeout-ms M             an actor, request or inference lane stalled this long is a failure\n"
                     "  --shutdown-timeout-ms M    longest shutdown waits for native calls before exiting nonzero\n"
                     "  --telemetry FILE.jsonl     actor, queue and lane counters, about once a second\n"
                     "  --sims S                   Gumbel search simulations per decision\n"
                     "  --particles P              worlds sampled for what the player cannot see; each simulation plays one\n"
                     "  --gumbel-scale G           1 explores (self-play); 0 is deterministic (evaluation)\n"
                     "  --rollout-mix M            in [0, 1]: weight of a guided rollout's outcome in each leaf value\n";
        return 2;
    }
    std::map<std::string, std::string> options;  // every option is required, once
    for (int i = 4; i + 1 < argc; i += 2)
        if (!options.emplace(argv[i], argv[i + 1]).second) throw std::invalid_argument{std::string{"repeated option "} + argv[i]};
    const auto take = [&options](const std::string& name) {
        const auto found = options.find(name);
        if (found == options.end()) throw std::invalid_argument{"missing option " + name};
        const auto value = found->second;
        options.erase(found);
        return value;
    };
    const int threads = std::stoi(take("--threads")), parallel = std::stoi(take("--parallel"));
    const int gpu_wait_us = std::stoi(take("--gpu-wait-us")), timeout_ms = std::stoi(take("--timeout-ms"));
    const int shutdown_timeout_ms = std::stoi(take("--shutdown-timeout-ms"));
    const auto telemetry = take("--telemetry");
    const stsrl::search::SearchConfig search{std::stoi(take("--sims")), std::stoi(take("--particles")),
                                             std::stod(take("--gumbel-scale")), std::stod(take("--rollout-mix"))};
    if (!options.empty()) throw std::invalid_argument{"unknown option " + options.begin()->first};
    if (threads < 1 || parallel < 1 || gpu_wait_us < 0 || timeout_ms <= 0 || shutdown_timeout_ms <= 0)
        throw std::invalid_argument{"positive threads/slots/timeouts and nonnegative wait required"};
    const auto setup_begin = Clock::now();
    const auto network = load_network(argv[2]);
    std::vector<stsrl::infra::Backend*> backends;
    for (const auto& backend : network.backends) backends.push_back(backend.get());
    std::cerr << "timing: evaluator setup "
              << std::chrono::duration<double>(Clock::now() - setup_begin).count() << " s\n";
    Shared shared{search.factory(), NetworkEvaluator{network.encoding}, {}, {}, {},
                  std::ofstream{argv[3], std::ios::binary | std::ios::app | std::ios::ate},
                  std::filesystem::path{argv[3]}.filename().string()};
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) shared.fights.push_back(Json::parse(line));
    const auto play_begin = Clock::now();
    const double play_wall_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    stsrl::infra::ActorConfig config;
    config.actors = threads; config.slots = parallel;
    config.batch_wait = std::chrono::microseconds{gpu_wait_us};
    config.timeout = std::chrono::milliseconds{timeout_ms};
    config.shutdown_timeout = std::chrono::milliseconds{shutdown_timeout_ms};
    stsrl::infra::TelemetryFile telemetry_file{telemetry};
    config.on_counters = [&](const stsrl::infra::Counters& counters) { telemetry_file.write(counters); };
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
