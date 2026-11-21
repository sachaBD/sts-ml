#include "agents/combat/nn/training/sample.hpp"
#include "agents/combat/player/decision.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include "environments/combat/state_codec.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>

namespace {
using namespace stsrl;
using Json = nlohmann::json;
struct Row {
    std::string state;
    std::vector<search::Action> actions;
    std::vector<float> policy;
    player::Vitals vitals;
    std::array<infra::Tokens, 2> tokens;
};

void legacy(std::ostream& out, const infra::Tokens& tokens, const std::vector<float>& policy,
            const player::Targets& targets) {
    const std::int32_t header[]{tokens.count, tokens.actions, int(tokens.data.size() / tokens.count)};
    out.write(reinterpret_cast<const char*>(header), sizeof header);
    out.write(reinterpret_cast<const char*>(targets.data()), sizeof targets);
    out.write(reinterpret_cast<const char*>(tokens.data.data()), std::streamsize(tokens.data.size() * sizeof(float)));
    out.write(reinterpret_cast<const char*>(policy.data()), std::streamsize(policy.size() * sizeof(float)));
}

void generate(const std::filesystem::path& root) {
    const char* names[]{"state", "entity", "inventory_v1"};
    std::array<std::ofstream, 3> samples, results;
    for (int i = 0; i < 3; ++i) {
        std::filesystem::create_directories(root / names[i]);
        samples[i].open(root / names[i] / "samples-0.bin", std::ios::binary);
        results[i].open(root / names[i] / "results.jsonl");
    }
    bool nested_choice = false;
    for (std::uint64_t seed = 0; seed < 3; ++seed) {
        auto start = scenarios::jaw_worm(seed + 1);
        if (seed == 0) {
            start.cards = sts::CardManager{};
            start.cards.createTempCardInHand(sts::CardInstance{sts::CardId::HAVOC});
            start.cards.createTempCardInHand(sts::CardInstance{sts::CardId::STRIKE_RED});
            start.cards.createTempCardInDrawPile(0, sts::CardInstance{sts::CardId::HEADBUTT});
            start.cards.createTempCardInDiscard(sts::CardInstance{sts::CardId::DEFEND_RED});
            start.cards.createTempCardInDiscard(sts::CardInstance{sts::CardId::BASH});
        }
        player::Fight fight{start, seed};
        std::mt19937_64 random{seed};
        std::vector<Row> rows;
        while (!fight.over()) {
            nested_choice |= fight.state().resolution.size() > 1;
            Row row;
            row.actions = combat::legal_actions(fight.state());
            std::ranges::reverse(row.actions);  // policy alignment must not depend on enumeration order
            row.policy.resize(row.actions.size());
            std::iota(row.policy.begin(), row.policy.end(), 1.f);
            const float sum = std::accumulate(row.policy.begin(), row.policy.end(), 0.f);
            for (auto& p : row.policy) p /= sum;
            row.vitals = player::vitals(fight.state());
            combat::pack(fight.state(), row.state);
            row.tokens = {nn::encode(fight.state(), row.actions, nn::Encoding::entity),
                          nn::encode(fight.state(), row.actions, nn::Encoding::inventory_v1)};
            auto action = row.actions[random() % row.actions.size()];
            if (seed == 0 && rows.empty()) {
                const auto found = std::ranges::find_if(row.actions, [&](const auto a) {
                    return a.getActionType() == sts::search::ActionType::CARD &&
                           fight.state().cards.hand[a.getSourceIdx()].id == sts::CardId::HAVOC;
                });
                if (found == row.actions.end()) throw std::runtime_error{"fixture Havoc not playable"};
                action = *found;
            }
            rows.push_back(std::move(row));
            fight.play(action);
        }
        std::array<std::streamoff, 3> begin;
        for (int i = 0; i < 3; ++i) begin[i] = samples[i].tellp();
        for (const auto& row : rows) {
            const auto targets = player::targets(row.vitals, fight.turn_starts(), fight.state());
            nn::training::write_sample(samples[0], row.state, row.actions, row.policy, targets);
            for (int i = 0; i < 2; ++i) legacy(samples[i + 1], row.tokens[i], row.policy, targets);
        }
        for (int i = 0; i < 3; ++i) {
            results[i] << Json{{"fight_id", "fixture:1:deck" + std::to_string(seed) + ":0"},
                               {"won", fight.state().outcome == sts::Outcome::PLAYER_VICTORY},
                               {"decisions", rows.size()}, {"sample_format", i ? 3 : nn::layout::sample::format},
                               {"sample_range", {{"file", "samples-0.bin"}, {"begin", begin[i]},
                                                  {"end", std::streamoff(samples[i].tellp())}}}}.dump() << '\n';
            if (!samples[i] || !results[i]) throw std::runtime_error{"fixture write failed"};
        }
    }
    if (!nested_choice) throw std::runtime_error{"fixture missed a suspended nested choice"};
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument{"expected output directory"};
    generate(argv[1]);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
