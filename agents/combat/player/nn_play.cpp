// nn_play: a separate process per play stage (see readme.md). Plays fight starts with search and the network.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/episodes.hpp"
#include "agents/combat/search/config.hpp"
#include "infra/workers/actors.hpp"
#include "infra/workers/telemetry.hpp"
#include <CLI/CLI.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) try {
    CLI::App app{"Plays the fight starts on stdin with search and the network. Appends each fight's samples to SAMPLES "
                 "and prints one result line to stdout. MODELS is an exported model directory: its network.json names the "
                 "encoding, the GPU deployments (smallest token capacity first) and the CPU model, with their lanes. Each "
                 "input goes to the smallest deployment that fits it, else the CPU.", "nn_play"};
    std::filesystem::path models, samples, telemetry_path;
    stsrl::infra::ActorConfig config;
    int wait_us, timeout_ms, shutdown_timeout_ms;
    stsrl::search::SearchConfig search;
    app.add_option("MODELS", models, "exported model directory")->required();
    app.add_option("SAMPLES", samples, "sample file to append to")->required();
    app.add_option("--threads", config.actors, "actors (CPU threads playing fights)")->required();
    app.add_option("--parallel", config.slots, "fights in flight per actor")->required();
    app.add_option("--gpu-wait-us", wait_us, "longest a partial inference batch waits for more requests")->required();
    app.add_option("--timeout-ms", timeout_ms, "an actor, request or inference lane stalled this long is a failure")->required();
    app.add_option("--shutdown-timeout-ms", shutdown_timeout_ms, "longest shutdown waits for native calls before exiting nonzero")->required();
    app.add_option("--telemetry", telemetry_path, "actor, queue and lane counters as JSON lines, about once a second")->required();
    app.add_option("--sims", search.simulations, "Gumbel search simulations per decision")->required();
    app.add_option("--particles", search.particles, "worlds sampled for what the player cannot see; each simulation plays one")->required();
    app.add_option("--gumbel-scale", search.gumbel_scale, "1 explores (self-play); 0 is deterministic (evaluation)")->required();
    app.add_option("--rollout-mix", search.rollout_mix, "in [0, 1]: weight of a guided rollout's outcome in each leaf value")->required();
    CLI11_PARSE(app, argc, argv);
    config.batch_wait = std::chrono::microseconds{wait_us};
    config.timeout = std::chrono::milliseconds{timeout_ms};
    config.shutdown_timeout = std::chrono::milliseconds{shutdown_timeout_ms};
    stsrl::infra::TelemetryFile telemetry{telemetry_path};
    config.on_counters = [&](const stsrl::infra::Counters& counters) { telemetry.write(counters); };

    const auto setup_begin = Clock::now();
    const auto network = stsrl::nn::load_network(models);
    std::vector<stsrl::infra::Backend*> backends;
    for (const auto& backend : network.backends) backends.push_back(backend.get());
    std::cerr << "timing: evaluator setup " << std::chrono::duration<double>(Clock::now() - setup_begin).count() << " s\n";
    std::vector<Json> fights;
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) fights.push_back(Json::parse(line));
    const auto count = fights.size();
    stsrl::player::Episodes episodes{std::move(fights), search.factory(), stsrl::nn::NetworkEvaluator{network.encoder},
                                     samples, std::cout};

    const auto play_begin = Clock::now();
    const double play_wall_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    try {
        stsrl::infra::run_actors(config, [&] { return episodes.next(); }, backends);
    } catch (const std::exception& e) {
        if (std::string{e.what()}.find("CUDA") != std::string::npos) {
            // A poisoned CUDA context requires a fresh process, not backend teardown.
            std::cerr << "nn_play: CUDA failure; exiting without GPU teardown: " << e.what() << '\n';
            std::fflush(nullptr);
            std::_Exit(1);
        }
        throw;
    }
    if (episodes.completed() != count) throw std::runtime_error{"missing completed fights"};
    const double seconds = std::chrono::duration<double>(Clock::now() - play_begin).count();
    std::cerr << "timing: gameplay " << seconds << " s\n";
    std::cerr << "play_stats: " << Json{{"seconds", seconds}, {"wall_start_unix", play_wall_start}, {"games", episodes.completed()},
        {"searched_decisions", episodes.decisions()}, {"decisions_per_second", episodes.decisions() / seconds},
        {"games_per_hour", episodes.completed() * 3600. / seconds}}.dump() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "nn_play: " << e.what() << "\n";
    return 1;
}
