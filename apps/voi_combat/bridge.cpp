// JSON-lines adapter: sample and score the requested experiment; Python chooses what to buy and play.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include "constants/MonsterEncounters.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace stsrl;
using Json = nlohmann::json;

namespace {
Json command() {
    std::string line;
    if (!std::getline(std::cin, line)) throw std::runtime_error{"unexpected end of decision"};
    return Json::parse(line);
}
void reply(const Json& j) { std::cout << j.dump() << std::endl; }

std::string describe(combat::Action action, const sts::BattleContext& state) {
    std::ostringstream text;
    action.printDesc(text, state);
    return text.str();
}

struct Scorer {
    nn::Network& net;
    player::Answers cache; // exact network input, scoped to one fight, same for both players
    int requests = 0, gpu_requests = 0;
    infra::Prediction eval(const sts::BattleContext& s, const std::vector<combat::Action>& actions) {
        auto tokens = nn::encode(s, actions, net.encoding);
        if (const auto* previous = cache.find(tokens)) return *previous;
        const auto backend = std::ranges::find_if(net.backends, [&](const auto& b){ return b->accepts(tokens); });
        if (backend == net.backends.end()) throw std::logic_error{"no backend accepts input"};
        std::vector<std::uint16_t> packed;
        (*backend)->prepare(tokens, packed);
        const std::array batch{infra::Input{&tokens, packed}};
        auto p = (*backend)->evaluate(batch).at(0);
        gpu_requests += backend->get() != net.backends.back().get();
        cache.add(tokens, p);
        ++requests;
        return p;
    }

    double rollout(const sts::BattleContext& world, std::uint64_t key, int elapsed, Json* path, const char*& endpoint) {
        player::Fight::ActionObserver observer;
        if (path) observer = [path](const auto& state, auto action, bool forced) {
            path->push_back({{"action", describe(action, state)}, {"forced", forced}});
        };
        player::Fight f{world, 0, std::move(observer)};
        auto actions = combat::legal_actions(f.state());
        auto move = std::ranges::find_if(actions, [&](auto a){ return combat::action_key(f.state(), a) == key; });
        if (move == actions.end()) throw std::runtime_error{"public move missing from particle"};
        f.play(*move);
        while (!f.over() && elapsed + f.steps() < player::Fight::max_steps) {
            actions = combat::legal_actions(f.state());
            const auto p = eval(f.state(), actions);
            if (f.state().turn != world.turn) { endpoint = "Next turn"; return p.value; }
            f.play(actions.at(std::ranges::max_element(p.logits) - p.logits.begin()));
        }
        endpoint = f.state().outcome == sts::Outcome::UNDECIDED ? "Limit" : "Terminal";
        return double(f.state().outcome == sts::Outcome::PLAYER_VICTORY);
    }
};
}

int main(int argc, char** argv) try {
    bool gpu = false, trace = false;
    if (argc < 2) throw std::invalid_argument{"usage: voi_combat_bridge MODEL_DIR [--gpu] [--trace]"};
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        if (flag == "--gpu") gpu = true;
        else if (flag == "--trace") trace = true;
        else throw std::invalid_argument{"unknown bridge option"};
    }
    auto net = nn::load_network(argv[1], gpu ? nn::Backends::all : nn::Backends::cpu);
    for (std::string line; std::getline(std::cin, line);) {
        const auto start = Json::parse(line);
        const auto seed = start.at("seed").get<std::uint64_t>();
        player::Fight fight{combat::start_battle(start), seed};
        Scorer scorer{net, {}};
        while (!fight.over()) {
            const auto& s = fight.state();
            auto actions = combat::legal_actions(s);
            const int before_root = scorer.requests;
            const auto p = scorer.eval(s, actions);
            Json root{{"logits", p.logits}, {"value", p.value}, {"step", fight.steps()}};
            if (trace) {
                root["requests"] = scorer.requests - before_root;
                root["turn"] = s.turn;
                root["actions"] = Json::array();
                for (auto action : actions) root["actions"].push_back(describe(action, s));
            }
            reply(root);
            // Separate from the true fight RNG; each probe buys a fresh world shared only within its pair.
            std::seed_seq probe_seed{std::uint32_t(seed), std::uint32_t(seed >> 32), std::uint32_t(fight.steps()), 1u};
            std::mt19937_64 random{probe_seed};
            while (true) {
                auto c = command();
                if (c.contains("play")) { fight.play(actions.at(c.at("play").get<size_t>())); break; }
                auto pair = c.at("pair").get<std::array<size_t, 2>>();
                const auto world = combat::particles(s, 1, random()).front();
                const int before = scorer.requests;
                std::array<double, 2> values{};
                Json branches = Json::array();
                for (std::size_t i = 0; i < pair.size(); ++i) {
                    Json path = Json::array();
                    const char* endpoint;
                    const int requests = scorer.requests;
                    values[i] = scorer.rollout(world, combat::action_key(s, actions.at(pair[i])), fight.steps(),
                                               trace ? &path : nullptr, endpoint);
                    if (trace) branches.push_back({{"action", pair[i]}, {"value", values[i]}, {"path", std::move(path)},
                                                   {"end", endpoint}, {"requests", scorer.requests-requests}});
                }
                Json result{{"difference", values[0]-values[1]}, {"requests", scorer.requests-before}};
                if (trace) result["branches"] = std::move(branches);
                reply(result);
            }
        }
        Json result{{"win", fight.state().outcome == sts::Outcome::PLAYER_VICTORY}, {"requests", scorer.requests},
                    {"gpu_requests", scorer.gpu_requests}, {"capped", fight.state().outcome == sts::Outcome::UNDECIDED}};
        if (trace) result["name"] = sts::monsterEncounterStrings[start.at("encounter").get<std::size_t>()];
        reply(result);
    }
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
