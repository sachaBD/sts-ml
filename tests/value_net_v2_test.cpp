// Parity of the C++ ValueNet (models/value_net.cpp) with PyTorch DeepSetsValueV2.
//   value_net_v2_test GOLDEN_DIR NAME...   GOLDEN_DIR/NAME.{bin,json} from tests/value_net_v2_golden.py
// States: tests/data/value_net_v2_states.json (real combat_v3 rows). Max |C++ - PyTorch| must be <= 1e-5, on a
// cold pass and on a pass served from the token caches. Run by tests/test_value_net_v2.py.
#include "models/value_net.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace stsrl;
using Json = nlohmann::json;

namespace {

template <std::size_t N>
void numeric(const Json& j, std::array<float, N>& out) {
    if (j.size() != N) throw std::runtime_error{"numeric size"};
    for (std::size_t i = 0; i < N; ++i) out[i] = j[i].get<float>();
}

EncodedCombatState parse(const Json& j) {
    EncodedCombatState s;
    numeric(j.at("global_numeric"), s.global.numeric);
    s.global.input_state = j.at("input_state").get<int>();
    s.global.card_selection_task = j.at("card_selection_task").get<int>();
    for (const auto& c : j.at("cards")) {
        CardToken t;
        t.card_id = c.at("card_id").get<int>();
        t.zone = static_cast<CardZone>(c.at("zone").get<int>());
        t.card_type = static_cast<CardType>(c.at("card_type").get<int>());
        t.target_type = static_cast<TargetType>(c.at("target_type").get<int>());
        numeric(c.at("numeric"), t.numeric);
        s.cards.push_back(t);
    }
    for (const auto& m : j.at("monsters")) {
        MonsterToken t;
        t.monster_id = m.at("monster_id").get<int>();
        t.move_id = m.at("move_id").get<int>();
        numeric(m.at("numeric"), t.numeric);
        s.monsters.push_back(t);
    }
    for (const auto& i : j.at("card_monster_interactions")) {
        CardMonsterInteraction t;
        t.card_index = i.at("card_index").get<std::uint16_t>();
        t.monster_index = i.at("monster_index").get<std::uint8_t>();
        numeric(i.at("numeric"), t.numeric);
        s.card_monster_interactions.push_back(t);
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: value_net_v2_test GOLDEN_DIR NAME...\n";
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    const std::filesystem::path states_path = std::filesystem::path{STSRL_SOURCE_DIR} / "tests/data/value_net_v2_states.json";
    std::vector<EncodedCombatState> states;
    for (const auto& j : Json::parse(std::ifstream{states_path})) states.push_back(parse(j));
    bool ok = true;
    for (int a = 2; a < argc; ++a) {
        const std::string name = argv[a];
        const auto golden = Json::parse(std::ifstream{dir / (name + ".json")});
        const auto expected = golden.at("values").get<std::vector<float>>();
        if (expected.size() != states.size()) throw std::runtime_error{"golden size mismatch"};
        const ValueNet net{(dir / (name + ".bin")).string()};
        double worst = 0;
        std::vector<float> values;
        for (int pass = 0; pass < 2; ++pass) {  // second pass: token encodings from the caches
            net.evaluate(states, values);
            for (std::size_t i = 0; i < states.size(); ++i)
                worst = std::max(worst, static_cast<double>(std::fabs(values[i] - expected[i])));
        }
        const bool pass = worst <= 1e-5;
        ok &= pass;
        std::printf("%-14s %zu states  max_abs_diff %.3g  %s\n", name.c_str(), states.size(), worst, pass ? "ok" : "FAIL");
    }
    return ok ? 0 : 1;
}
