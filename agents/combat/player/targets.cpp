#include "agents/combat/player/targets.hpp"
#include <algorithm>
#include <stdexcept>

namespace stsrl::player {
Vitals vitals(const sts::BattleContext& state) {
    Vitals v{state.turn, state.player.curHp, state.player.maxHp, 0, 0};
    for (int i = 0; i < state.monsters.monsterCount; ++i) {
        v.enemy_hp += std::max(0, state.monsters.arr[i].curHp);
        v.enemy_max_hp += state.monsters.arr[i].maxHp;
    }
    if (v.max_hp <= 0 || v.enemy_max_hp <= 0) throw std::runtime_error{"targets: fight without player or enemy HP"};
    return v;
}

void observe(std::vector<Vitals>& turn_starts, const sts::BattleContext& state) {
    if (turn_starts.empty() || turn_starts.back().turn != state.turn) turn_starts.push_back(vitals(state));
}

Targets targets(const Vitals& decision, std::span<const Vitals> turn_starts, const sts::BattleContext& end) {
    const auto last = vitals(end);
    const auto next = std::find_if(turn_starts.begin(), turn_starts.end(),
                                   [&](const Vitals& v) { return v.turn > decision.turn; });
    const auto& round = next == turn_starts.end() ? last : *next;
    namespace target = nn::layout::sample::targets;
    Targets t;
    t[target::won] = float(end.outcome == sts::Outcome::PLAYER_VICTORY);
    t[target::hp] = float(last.hp) / float(last.max_hp);
    t[target::enemy_hp] = float(last.enemy_hp) / float(last.enemy_max_hp);
    t[target::hp_lost] = float(decision.hp - round.hp) / float(decision.max_hp);
    t[target::enemy_hp_lost] = float(decision.enemy_hp - round.enemy_hp) / float(decision.enemy_max_hp);
    t[target::turns_remaining] = float(last.turn - decision.turn);
    return t;
}
}  // namespace stsrl::player
