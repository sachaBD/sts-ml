#include "agents/combat/pv/features.hpp"
#include "environments/combat/environment.hpp"
#include <algorithm>
#include <stdexcept>

namespace stsrl::pv {
namespace {
void card(std::vector<float>& out, const CardToken& c) {
    out.insert(out.end(), {float(c.card_id), float(c.zone), float(c.card_type), float(c.target_type)});
    out.insert(out.end(), c.numeric.begin(), c.numeric.end());
}
void monster(std::vector<float>& out, MonsterToken m, const sts::BattleContext& bc) {
    // Frozen v4 exposes opaque simulator counters. Do not feed those into the new public-information model.
    m.status[13] = m.status[14] = 0;
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) {
        m.move_id = m.previous_move_id = 0;
        m.numeric[3] = m.numeric[4] = m.status[12] = 0;
    }
    out.insert(out.end(), {float(m.monster_id), float(m.move_id), float(m.previous_move_id)});
    out.insert(out.end(), m.numeric.begin(), m.numeric.end());
    out.insert(out.end(), m.status.begin(), m.status.end());
    float stances = 0, phase = 0;
    for (const auto& raw : bc.monsters.arr)
        if (raw.id == sts::MonsterId::THE_CHAMP && int(raw.id) == m.monster_id) {
            stances = float(raw.miscInfo & 3); phase = float((raw.miscInfo >> 2) & 1);
        }
    // These counters advance while selecting the next intent. With Dome they can reveal that hidden intent.
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) stances = phase = 0;
    out.insert(out.end(), {stances, phase});
}
void potion(std::vector<float>& out, const PotionToken& p) {
    out.push_back(float(p.potion_id));
    out.insert(out.end(), p.numeric.begin(), p.numeric.end());
}
}  // namespace

Inputs encode(const sts::BattleContext& bc, std::span<const sts::search::Action> moves) {
    if (bc.player.cc != sts::CharacterClass::IRONCLAD)
        throw std::invalid_argument{"PV supports Ironclad only"};
    const auto state = encode_state(bc, CardCoverage::ironclad_events);
    Inputs out;
    auto& context = out[0];
    context.assign(state.global.numeric.begin(), state.global.numeric.end());
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) context[16] = context[17] = 0;
    context.insert(context.end(), state.global.player.begin(), state.global.player.end());
    context.insert(context.end(), {float(state.global.input_state), float(state.global.card_selection_task),
                                   state.global.max_hp / 100.f});
    for (const auto& c : state.cards) card(out[1], c);
    for (const auto& m : state.monsters) monster(out[2], m, bc);
    for (const auto& p : state.potions) potion(out[3], p);
    for (const auto& r : state.relics) {
        out[4].push_back(float(r.relic_id));
        out[4].insert(out[4].end(), r.numeric.begin(), r.numeric.end());
    }
    for (const auto& move : moves) {
        using T = sts::search::ActionType;
        if (move.getActionType() > T::END_TURN)
            throw std::invalid_argument{"PV: unsupported action type"};
        if (move.getActionType() == T::MULTI_CARD_SELECT &&
            bc.cardSelectInfo.cardSelectTask != sts::CardSelectTask::EXHAUST_MANY &&
            bc.cardSelectInfo.cardSelectTask != sts::CardSelectTask::GAMBLE)
            throw std::invalid_argument{"PV: unsupported subset selection task"};
        const auto a = encode_action(bc, move.bits, 0, CardCoverage::ironclad_events);
        std::vector<float> row{float(a.kind), float(a.card_selection_task), float(a.skips_selection),
                              float(a.discards_potion), float(a.source_card.has_value()),
                              float(a.target_monster.has_value()), float(a.potion.has_value()),
                              float(a.interaction.has_value())};
        if (a.source_card) card(row, *a.source_card); else row.resize(26, 0);
        if (a.target_monster) monster(row, *a.target_monster, bc); else row.resize(55, 0);
        if (a.potion) potion(row, *a.potion); else row.resize(74, 0);
        if (a.interaction) row.insert(row.end(), a.interaction->begin(), a.interaction->end());
        else row.resize(80, 0);
        if (move.getActionType() == T::MULTI_CARD_SELECT)
            for (int i : move.getSelectedIdxs()) {
                const sts::search::Action selection{T::SINGLE_CARD_SELECT, i};
                const auto token = encode_action(bc, selection.bits, 0, CardCoverage::ironclad_events);
                if (!token.source_card) throw std::runtime_error{"PV: subset card missing"};
                card(row, *token.source_card);
            }
        row.resize(260, 0); row.push_back(1);  // real action, not padding
        out[5].insert(out[5].end(), row.begin(), row.end());
    }
    // One zero token for an empty set, so the runtime never needs zero-sized dimensions.
    for (int i = 1; i < 6; ++i) if (out[i].empty()) out[i].resize(widths[i], 0);
    return out;
}
}  // namespace stsrl::pv
