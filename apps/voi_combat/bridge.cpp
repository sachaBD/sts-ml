// JSON-lines adapter: sample and score the requested experiment; Python chooses what to buy and play.
#include "agents/combat/nn/network.hpp"
#include "agents/combat/player/decision.hpp"
#include "environments/combat/fight_start.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <array>
#include <iostream>
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

    double rollout(const sts::BattleContext& world, std::uint64_t key, int elapsed) {
        player::Fight f{world, 0};
        auto actions = combat::legal_actions(f.state());
        auto move = std::ranges::find_if(actions, [&](auto a){ return combat::action_key(f.state(), a) == key; });
        if (move == actions.end()) throw std::runtime_error{"public move missing from particle"};
        f.play(*move);
        while (!f.over() && elapsed + f.steps() < player::Fight::max_steps) {
            actions = combat::legal_actions(f.state());
            const auto p = eval(f.state(), actions);
            if (f.state().turn != world.turn) return p.value;
            f.play(actions.at(std::ranges::max_element(p.logits) - p.logits.begin()));
        }
        return double(f.state().outcome == sts::Outcome::PLAYER_VICTORY);
    }
};
}

int main(int argc, char** argv) try {
    const bool gpu = argc == 3 && std::string_view(argv[2]) == "--gpu";
    if (!gpu && argc != 2) throw std::invalid_argument{"usage: voi_combat_bridge MODEL_DIR [--gpu]"};
    auto net = nn::load_network(argv[1], gpu ? nn::Backends::all : nn::Backends::cpu);
    for (std::string line; std::getline(std::cin, line);) {
        const auto start = Json::parse(line);
        player::Fight fight{combat::start_battle(start), start.at("seed").get<std::uint64_t>()};
        Scorer scorer{net, {}};
        while (!fight.over()) {
            const auto& s = fight.state();
            auto actions = combat::legal_actions(s);
            const auto p = scorer.eval(s, actions);
            reply({{"logits", p.logits}, {"value", p.value}, {"step", fight.steps()}});
            // Separate from the true fight RNG; each probe buys a fresh world shared only within its pair.
            auto random = fight.decision_random(player::Fight::RandomStream::thinking_probe);
            while (true) {
                auto c = command();
                if (c.contains("play")) { fight.play(actions.at(c.at("play").get<size_t>())); break; }
                auto pair = c.at("pair").get<std::array<size_t, 2>>();
                const auto world = combat::particles(s, 1, random()).front();
                const int before = scorer.requests;
                const double a = scorer.rollout(world, combat::action_key(s, actions.at(pair[0])), fight.steps());
                const double b = scorer.rollout(world, combat::action_key(s, actions.at(pair[1])), fight.steps());
                reply({{"difference", a-b}, {"requests", scorer.requests-before}});
            }
        }
        reply({{"win", fight.state().outcome == sts::Outcome::PLAYER_VICTORY}, {"requests", scorer.requests},
               {"gpu_requests", scorer.gpu_requests}, {"capped", fight.state().outcome == sts::Outcome::UNDECIDED}});
    }
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
