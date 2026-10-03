// Rebuild combat_v4 fights with sts_lightspeed only and check them (the core of the combat_v4_full decompressor).
//
//   combat_v4_replay < fights.jsonl
//
// stdin: one combat_v4 fights row per line as JSON (start, actions, won, final_hp; other fields ignored).
// Each fight: combat_v4::replay, then won / final_hp must match. Exits non-zero on the first mismatch.
// stdout: {"fights", "decisions", "parse_seconds", "replay_seconds"}.
#include "environments/combat/record_v4.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using Clock = std::chrono::steady_clock;

int main() {
    std::ios::sync_with_stdio(false);
    double parse_s = 0, replay_s = 0;
    std::int64_t fights = 0, decisions = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto t = Clock::now();
        const auto row = nlohmann::json::parse(line);
        const auto actions = row.at("actions").get<std::vector<std::uint32_t>>();
        parse_s += std::chrono::duration<double>(Clock::now() - t).count();
        t = Clock::now();
        const auto end = stsrl::combat_v4::replay(row.at("start"), actions);
        replay_s += std::chrono::duration<double>(Clock::now() - t).count();
        const bool won = end.outcome == sts::Outcome::PLAYER_VICTORY;
        if (won != row.at("won").get<bool>() || end.player.curHp != row.at("final_hp").get<int>()) {
            std::cerr << "mismatch: " << row.at("fight_id") << '\n';
            return 1;
        }
        ++fights;
        decisions += static_cast<std::int64_t>(actions.size());
    }
    std::cout << nlohmann::json{{"fights", fights}, {"decisions", decisions}, {"parse_seconds", parse_s},
                                {"replay_seconds", replay_s}}.dump() << '\n';
    return 0;
}
