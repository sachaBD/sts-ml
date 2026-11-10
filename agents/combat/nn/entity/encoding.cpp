#include "agents/combat/nn/entity/encoding.hpp"
#include "agents/combat/nn/features.hpp"
#include <stdexcept>

namespace stsrl::nn {
namespace {
namespace L = layout::entity;
// The C++ enums this encoding writes as values must number them as the layout does.
static_assert(int(CardZone::hand) == L::card_zone::hand && int(CardZone::draw) == L::card_zone::draw &&
              int(CardZone::discard) == L::card_zone::discard && int(CardZone::exhaust) == L::card_zone::exhaust &&
              int(CardZone::offered) == L::card_zone::offered);
static_assert(int(EncodedActionKind::card) + 1 == L::action_kind::card &&
              int(EncodedActionKind::potion) + 1 == L::action_kind::potion &&
              int(EncodedActionKind::single_card_selection) + 1 == L::action_kind::single_card_selection &&
              int(EncodedActionKind::multi_card_selection) + 1 == L::action_kind::multi_card_selection &&
              int(EncodedActionKind::end_turn) + 1 == L::action_kind::end_turn);
static_assert(L::link::target == width - 1, "the target link is a row's last column");

struct Row {
    std::vector<float>& out;
    std::size_t start;
    Row(std::vector<float>& out, int kind, int id, int id2) : out{out}, start{out.size()} {
        out.resize(start + width, 0);
        out[start + layout::row::kind] = float(kind);
        out[start + layout::row::id] = float(id);
        out[start + layout::row::id2] = float(id2);
    }
    std::size_t at = layout::row::features;
    void put(float x) {
        if (at >= std::size_t(width)) throw std::logic_error{"nn: token overflow"};
        out[start + at++] = x;
    }
    template<class R> void put_all(const R& xs) { for (float x : xs) put(x); }
};

std::array<float, 17> card_features(const CardToken& c) {
    std::array<float, 17> f{float(c.zone), float(c.card_type), float(c.target_type)};
    std::copy(c.numeric.begin(), c.numeric.end(), f.begin() + 3);
    return f;
}
}  // namespace

stsrl::infra::Tokens encode_entity(const sts::BattleContext& bc, std::span<const sts::search::Action> actions) {
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) throw std::invalid_argument{"nn: Runic Dome hides intents"};
    std::vector<int> monster_slots;
    const auto s = encode_state(bc, &monster_slots);
    stsrl::infra::Tokens t;
    auto& out = t.data;
    {
        Row r{out, L::kind::context, 0, L::version};
        r.put_all(s.global.numeric); r.put_all(s.global.player);
        r.put(float(s.global.input_state)); r.put(float(s.global.card_selection_task)); r.put(s.global.max_hp / 100.f);
    }
    for (const auto& c : s.cards) { Row r{out, L::kind::card, c.card_id, 0}; r.put_all(card_features(c)); }
    for (auto m : s.monsters) {
        m.status[13] = m.status[14] = 0;  // opaque simulator counters
        float stances = 0, phase = 0;
        for (const auto& raw : bc.monsters.arr)
            if (raw.id == sts::MonsterId::THE_CHAMP && int(raw.id) == m.monster_id) {
                stances = float(raw.miscInfo & 3); phase = float((raw.miscInfo >> 2) & 1);
            }
        Row r{out, L::kind::monster, m.monster_id, m.move_id};
        r.put(float(m.previous_move_id)); r.put_all(m.numeric); r.put_all(m.status); r.put(stances); r.put(phase);
    }
    for (const auto& p : s.potions) { Row r{out, L::kind::potion, p.potion_id, 0}; r.put_all(p.numeric); }
    for (const auto& x : s.relics) { Row r{out, L::kind::relic, x.relic_id, 0}; r.put_all(x.numeric); }
    for (const auto& move : actions) {
        using T = sts::search::ActionType;
        const auto a = encode_action(bc, move);
        const int id = a.source_card ? a.source_card->card_id : a.potion_id.value_or(0);
        Row r{out, L::kind::action, id, int(a.kind) + 1};
        r.put(float(a.card_selection_task)); r.put(float(a.skips_selection)); r.put(float(a.discards_potion));
        r.put_all(a.interaction.value_or(std::array<float, 6>{}));
        r.put_all(a.source_card ? card_features(*a.source_card) : std::array<float, 17>{});
        std::array<float, 17> subset{};
        float selected = 0;
        if (move.getActionType() == T::MULTI_CARD_SELECT)
            for (int i : move.getSelectedIdxs()) {
                const auto c = encode_action(bc, sts::search::Action{T::SINGLE_CARD_SELECT, i});
                if (!c.source_card) throw std::runtime_error{"nn: subset card missing"};
                const auto f = card_features(*c.source_card);
                for (std::size_t k = 0; k < f.size(); ++k) subset[k] += f[k];
                ++selected;
            }
        r.put(selected); r.put_all(subset);
        r.put_all(a.potion ? a.potion->numeric : decltype(a.potion->numeric){});
        if (a.target_monster) {
            const auto target = std::find(monster_slots.begin(), monster_slots.end(), move.getTargetIdx());
            if (target == monster_slots.end()) throw std::logic_error{"nn: target monster missing"};
            const auto row = 1 + s.cards.size() + (target - monster_slots.begin());  // the monster's row
            out[r.start + L::link::target] = float(row + L::link_offset);
        }
    }
    t.count = int(out.size() / width);
    t.actions = int(actions.size());
    return t;
}
}  // namespace stsrl::nn
