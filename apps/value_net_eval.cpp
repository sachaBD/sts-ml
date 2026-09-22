// Scores encoded states with the native value net, for parity checks against PyTorch.
//   value_net_eval value_weights.bin states.jsonl
// Each input line is one state in the encoding's JSON form (as in the data rows); prints one
// value per line.
#include "models/value_net.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

stsrl::EncodedCombatState parse_state(const nlohmann::json& j) {
    stsrl::EncodedCombatState s;
    s.global.numeric = j.at("global_numeric").get<std::array<float, 50>>();
    s.global.input_state = j.at("input_state").get<int>();
    s.global.card_selection_task = j.at("card_selection_task").get<int>();
    for (const auto& c : j.at("cards"))
        s.cards.push_back({c.at("card_id").get<int>(), static_cast<stsrl::CardZone>(c.at("zone").get<int>()),
                           static_cast<stsrl::CardType>(c.at("card_type").get<int>()),
                           static_cast<stsrl::TargetType>(c.at("target_type").get<int>()),
                           c.at("numeric").get<std::array<float, 14>>()});
    for (const auto& m : j.at("monsters"))
        s.monsters.push_back({m.at("monster_id").get<int>(), m.at("move_id").get<int>(),
                              m.at("numeric").get<std::array<float, 9>>()});
    for (const auto& i : j.at("card_monster_interactions"))
        s.card_monster_interactions.push_back({i.at("card_index").get<std::uint16_t>(),
                                               i.at("monster_index").get<std::uint8_t>(),
                                               i.at("numeric").get<std::array<float, 6>>()});
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: value_net_eval value_weights.bin states.jsonl\n";
        return 2;
    }
    const stsrl::ValueNet net(argv[1]);
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) std::printf("%.9g\n", static_cast<double>(net.evaluate(parse_state(nlohmann::json::parse(line)))));
}
