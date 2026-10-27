// nn_play play MODEL.onnx SAMPLES.bin [--threads T] [--parallel G] [--gpu-model FIXED.onnx --gpu-batch B --gpu-tokens N]
//                                      [--sims S] [--rollout-mix M] [--gumbel-scale G] [--particles P]
//   Reads fights as JSON lines {"fight_id", "start"} on stdin and plays them with Gumbel search. T threads each keep
//   G games in flight, in two groups whose leaves are evaluated in batches: on the CPU, or with --gpu-model on the GPU
//   (FIXED.onnx = the same network exported for exactly [B, N, width]; see net.hpp).
//   Optional --gpu-bank ONNX,... supplies smaller batches of the SAME network and token shape;
//   --gpu-wait-us controls arrival-stall waiting. --groups / --scheduler control CPU overlap.
//   Prints one JSON line per fight
//   ({"fight_id", "won", "hp", "decisions", "seconds"}) and appends every searched decision to SAMPLES.bin.
//   A fight still undecided after 50 turns / 512 actions is a loss.
// nn_play import SAMPLES.bin
//   Reads recorded fights as JSON lines {"fight": {"start", "actions", "won"}, "search": [{"step", "children":
//   [{"action", "visits"}]}]} (e.g. MCTS games) and appends their searched decisions, policy = normalized visits.
// SAMPLES.bin record: int32 tokens, int32 actions, float32 won, float32 tokens x width, float32 policy[actions].
#include "agents/combat/nn/gumbel.hpp"
#include "agents/combat/nn/workers.hpp"
#include "environments/combat/record_v4.hpp"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <thread>

using namespace stsrl::nn;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
sts::BattleContext start(const Json& start) {
    sts::BattleContext state;
    state.init(stsrl::combat_v4::start_game(start));
    return state;
}

void write(std::ofstream& out, const Tokens& t, const std::vector<float>& policy, float won) {
    out.write(reinterpret_cast<const char*>(&t.count), 4);
    out.write(reinterpret_cast<const char*>(&t.actions), 4);
    out.write(reinterpret_cast<const char*>(&won), 4);
    out.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.data.size() * 4));
    out.write(reinterpret_cast<const char*>(policy.data()), std::streamsize(policy.size() * 4));
}

struct Game {
    Json fight;
    sts::BattleContext state;
    std::mt19937_64 rng;
    int steps = 0;
    std::optional<Search> search;
    std::vector<std::pair<Tokens, std::vector<float>>> decisions;  // (state, improved policy) per searched move
    Clock::time_point begin = Clock::now();

    explicit Game(Json f) : fight(std::move(f)), state{start(fight.at("start"))},
                            rng{fight.at("start").at("seed").get<std::uint64_t>()} {}

    bool over() const { return state.outcome != sts::Outcome::UNDECIDED || state.turn >= 50 || steps >= 512; }

    // Play on until the search needs the network (true) or the game is over (false).
    bool advance(const Settings& settings) {
        for (;;) {
            if (search) {
                if (search->pending()) return true;
                auto r = search->result();
                decisions.emplace_back(encode(state, r.actions), std::move(r.policy));
                r.action.execute(state); ++steps;
                search.reset();
            }
            if (over()) return false;
            const auto actions = legal_actions(state);
            if (actions.size() == 1) { actions[0].execute(state); ++steps; continue; }  // forced: nothing to learn
            search.emplace(state, settings, rng());
        }
    }
};

// Everything the worker threads share.
struct Shared {
    std::unique_ptr<Evaluator> evaluator;
    Settings settings;
    std::vector<Json> fights;
    std::atomic<std::size_t> next{0};
    std::mutex output;
    std::ofstream samples;
    std::atomic<std::size_t> completed{0}, searched_decisions{0};

    std::unique_ptr<Game> next_game() {
        const auto i = next++;
        return i < fights.size() ? std::make_unique<Game>(fights[i]) : nullptr;
    }

    void finish(const Game& g) {
        const float won = g.state.outcome == sts::Outcome::PLAYER_VICTORY;
        const double seconds = std::chrono::duration<double>(Clock::now() - g.begin).count();
        std::lock_guard lock{output};
        for (const auto& [tokens, policy] : g.decisions) write(samples, tokens, policy, won);
        samples.flush();
        std::cout << Json{{"fight_id", g.fight.at("fight_id")}, {"won", won > 0}, {"hp", g.state.player.curHp},
                          {"decisions", g.decisions.size()}, {"seconds", seconds}}.dump() << std::endl;
        if (!samples || !std::cout) throw std::runtime_error{"failed to write game samples/results"};
        ++completed;
        searched_decisions += g.decisions.size();
    }
};

// Adapter keeps the real simulator/search and output on their owning SimWorker.
class CombatSimulation final : public Simulation {
    std::unique_ptr<Game> game_;
    Shared& shared_;
public:
    CombatSimulation(std::unique_ptr<Game> game, Shared& shared) : game_{std::move(game)}, shared_{shared} {}
    const Tokens* advance() override {
        return game_->advance(shared_.settings) ? game_->search->pending() : nullptr;
    }
    void evaluated(const Prediction& prediction) override { game_->search->evaluated(prediction); }
    void finish() override { shared_.finish(*game_); }
    std::string label() const override { return game_->fight.at("fight_id").get<std::string>(); }
};

// A group of games whose pending leaves are evaluated together.
struct Group {
    std::vector<std::unique_ptr<Game>> games;
    std::vector<Game*> waiting;                       // games whose leaf is being evaluated, in batch order
    std::future<std::vector<Prediction>> evaluating;

    // Hand the evaluated leaves back to their searches.
    void receive() {
        if (!evaluating.valid()) return;
        const auto predictions = evaluating.get();
        for (std::size_t i = 0; i < waiting.size(); ++i) waiting[i]->search->evaluated(predictions[i]);
    }

    // Play every game on to its next leaf (finishing and replacing games that end) and submit the leaves.
    // False once the group has no games left.
    bool play_on(Shared& shared) {
        waiting.clear();
        std::vector<const Tokens*> leaves;
        for (auto& g : games) {
            while (g && !g->advance(shared.settings)) { shared.finish(*g); g = shared.next_game(); }
            if (g) { waiting.push_back(g.get()); leaves.push_back(g->search->pending()); }
        }
        if (leaves.empty()) return false;
        evaluating = shared.evaluator->submit(std::move(leaves));
        return true;
    }
};

// Several independent groups per worker overlap simulation with inference (two ordered groups by default).
void legacy_worker(Shared& shared, int parallel, int group_count, bool ready_first) {
    std::vector<Group> groups(group_count);
    for (int k = 0; k < group_count; ++k)
        for (int i = 0; i < parallel / group_count + (k < parallel % group_count); ++i)
            groups[k].games.push_back(shared.next_game());
    if (!ready_first) {
        for (bool active = true; active;) {
            active = false;
            for (auto& group : groups) {
                group.receive();
                active |= group.play_on(shared);
            }
        }
        return;
    }
    // Do not block on one group's future while another group already has usable predictions.
    for (auto& group : groups) group.play_on(shared);
    for (;;) {
        bool pending = false, progressed = false;
        for (auto& group : groups) {
            if (!group.evaluating.valid()) continue;
            pending = true;
            if (group.evaluating.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
                group.receive();
                group.play_on(shared);
                progressed = true;
            }
        }
        if (!pending) return;
        if (!progressed) {
            // Bounded sleep rather than spinning a core while every group waits on inference.
            for (auto& group : groups) if (group.evaluating.valid()) {
                group.evaluating.wait_for(std::chrono::microseconds{50});
                break;
            }
        }
    }
}

void import(std::ofstream& samples) {
    using Public = sts::search::PublicBeliefCombatSearch;
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        const auto record = Json::parse(line);
        const auto& fight = record.at("fight");
        std::map<int, const Json*> searched;
        for (const auto& row : record.at("search")) searched[row.at("step").get<int>()] = &row;
        auto state = start(fight.at("start"));
        const float won = fight.at("won").get<bool>();
        int step = 0;
        for (const auto& bits : fight.at("actions")) {
            if (searched.contains(step)) {
                const auto actions = legal_actions(state);
                std::vector<float> policy(actions.size(), 0);
                float total = 0;
                for (const auto& child : searched[step]->at("children")) {
                    const auto key = Public::publicActionKey(state, sts::search::Action{child.at("action").get<std::uint32_t>()});
                    std::size_t i = 0;
                    while (i < actions.size() && Public::publicActionKey(state, actions[i]) != key) ++i;
                    if (i == actions.size()) throw std::runtime_error{"import: recorded action is not legal"};
                    policy[i] += child.at("visits").get<float>(); total += child.at("visits").get<float>();
                }
                if (actions.size() > 1 && total > 0) {
                    for (auto& p : policy) p /= total;
                    write(samples, encode(state, actions), policy, won);
                }
            }
            sts::search::Action{bits.get<std::uint32_t>()}.execute(state);
            ++step;
        }
        if ((state.outcome == sts::Outcome::PLAYER_VICTORY) != (won > 0)) throw std::runtime_error{"import: fight did not replay"};
    }
}
}  // namespace

int main(int argc, char** argv) try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "import" && argc == 3) {
        std::ofstream samples{argv[2], std::ios::binary | std::ios::app};
        import(samples);
        return 0;
    }
    if (mode != "play" || argc < 4 || argc % 2) {
        std::cerr << "usage: nn_play play MODEL.onnx SAMPLES.bin [--threads T] [--parallel G] [--gpu-model FIXED.onnx\n"
                     "                    --gpu-batch B --gpu-tokens N] [--gpu-bank SMALLER.onnx,...] [--gpu-wait-us U]\n"
                     "                    [--groups G] [--scheduler local|ordered|ready] [--sims S] [--rollout-mix M]\n"
                     "                    [--gumbel-scale G] [--particles P] [--timeout-ms M] [--shutdown-timeout-ms M]\n"
                     "       nn_play import SAMPLES.bin\n";
        return 2;
    }
    int threads = 10, parallel = 64, groups = 2, gpu_batch = 1024, gpu_tokens = 48, gpu_wait_us = 1000;
    int timeout_ms = 60'000, shutdown_timeout_ms = 5'000;
    std::string gpu_model, scheduler = "ordered";
    std::vector<std::filesystem::path> gpu_bank;
    Settings settings;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--threads") threads = std::stoi(v);
        else if (k == "--parallel") parallel = std::stoi(v);
        else if (k == "--groups") groups = std::stoi(v);
        else if (k == "--scheduler") scheduler = v;
        else if (k == "--timeout-ms") timeout_ms = std::stoi(v);
        else if (k == "--shutdown-timeout-ms") shutdown_timeout_ms = std::stoi(v);
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
        else if (k == "--gpu-tokens") gpu_tokens = std::stoi(v);
        else if (k == "--sims") settings.simulations = std::stoi(v);
        else if (k == "--rollout-mix") settings.rollout_mix = std::stod(v);
        else if (k == "--gumbel-scale") settings.gumbel_scale = std::stod(v);
        else if (k == "--particles") settings.particles = std::stoi(v);
        else throw std::invalid_argument{"unknown option " + k};
    }
    const bool local = scheduler == "local";
    if (threads < 1 || parallel < 1 || gpu_batch < 1 || gpu_wait_us < 0 || timeout_ms <= 0 || shutdown_timeout_ms <= 0 ||
        settings.simulations < 1 || settings.particles < 1 || settings.rollout_mix < 0 || settings.rollout_mix > 1 || settings.gumbel_scale < 0)
        throw std::invalid_argument{"positive threads/slots/batch/sims/particles/timeouts, rollout mix in [0,1], nonnegative noise/wait required"};
    if (!local && scheduler != "ordered" && scheduler != "ready") throw std::invalid_argument{"scheduler must be local, ordered or ready"};
    if (!local && (groups < 2 || parallel < groups)) throw std::invalid_argument{"legacy scheduler requires parallel >= groups >= 2"};
    if (!local && !gpu_model.empty() && (parallel + groups - 1) / groups > gpu_batch)
        throw std::invalid_argument{"each legacy worker group must fit within gpu-batch"};
    if (gpu_model.empty() && !gpu_bank.empty()) throw std::invalid_argument{"gpu-bank requires gpu-model"};
    const auto setup_begin = Clock::now();
    auto evaluator = gpu_model.empty() ? cpu_evaluator(argv[2]) : gpu_evaluator(argv[2], gpu_model, gpu_batch, gpu_tokens, gpu_bank, gpu_wait_us, local);
    std::cerr << "timing: evaluator setup "
              << std::chrono::duration<double>(Clock::now() - setup_begin).count() << " s\n";
    Shared shared{std::move(evaluator), settings, {}, {}, {}, std::ofstream{argv[3], std::ios::binary | std::ios::app}};
    for (std::string line; std::getline(std::cin, line);)
        if (!line.empty()) shared.fights.push_back(Json::parse(line));
    const auto play_begin = Clock::now();
    const double play_wall_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (local) {
        WorkerConfig config;
        config.workers = threads; config.slots = parallel; config.batch = gpu_batch;
        config.batch_wait = std::chrono::microseconds{gpu_wait_us};
        config.timeout = std::chrono::milliseconds{timeout_ms};
        config.shutdown_timeout = std::chrono::milliseconds{shutdown_timeout_ms};
        run_workers(config, [&]() -> std::unique_ptr<Simulation> {
            auto game = shared.next_game();
            return game ? std::make_unique<CombatSimulation>(std::move(game), shared) : nullptr;
        }, [&](std::vector<const Tokens*> tokens) { return shared.evaluator->submit(std::move(tokens)).get(); });
    } else {
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; ++t) pool.emplace_back(legacy_worker, std::ref(shared), parallel, groups, scheduler == "ready");
        pool.clear();
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
