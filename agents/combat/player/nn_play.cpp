// nn_play: a separate process per play stage (see readme.md). Plays fight starts with search and the network.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/episodes.hpp"
#include "agents/combat/search/config.hpp"
#include "infra/workers/actors.hpp"
#include "infra/workers/telemetry.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

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
    stsrl::infra::ActorConfig config;
    config.actors = std::stoi(take("--threads"));
    config.slots = std::stoi(take("--parallel"));
    config.batch_wait = std::chrono::microseconds{std::stoi(take("--gpu-wait-us"))};
    config.timeout = std::chrono::milliseconds{std::stoi(take("--timeout-ms"))};
    config.shutdown_timeout = std::chrono::milliseconds{std::stoi(take("--shutdown-timeout-ms"))};
    stsrl::infra::TelemetryFile telemetry{take("--telemetry")};
    config.on_counters = [&](const stsrl::infra::Counters& counters) { telemetry.write(counters); };
    const stsrl::search::SearchConfig search{std::stoi(take("--sims")), std::stoi(take("--particles")),
                                             std::stod(take("--gumbel-scale")), std::stod(take("--rollout-mix"))};
    if (!options.empty()) throw std::invalid_argument{"unknown option " + options.begin()->first};

    const auto setup_begin = Clock::now();
    const auto network = stsrl::nn::load_network(argv[2]);
    std::vector<stsrl::infra::Backend*> backends;
    for (const auto& backend : network.backends) backends.push_back(backend.get());
    std::cerr << "timing: evaluator setup " << std::chrono::duration<double>(Clock::now() - setup_begin).count() << " s\n";
    std::vector<Json> fights;
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) fights.push_back(Json::parse(line));
    const auto count = fights.size();
    stsrl::player::Episodes episodes{std::move(fights), search.factory(), stsrl::nn::NetworkEvaluator{network.encoding},
                                     argv[3], std::cout};

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
