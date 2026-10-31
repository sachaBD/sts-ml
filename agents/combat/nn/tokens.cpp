#include "agents/combat/nn/tokens.hpp"
#include "environments/combat/environment.hpp"
#include <stdexcept>

namespace stsrl::nn {
namespace {
enum Type { context = 1, card, monster, potion, relic, action };

struct Row {
    std::vector<float>& out;
    std::size_t start;
    Row(std::vector<float>& out, int type, int id, int id2) : out{out}, start{out.size()} {
        out.resize(start + width, 0);
        out[start] = float(type); out[start + 1] = float(id); out[start + 2] = float(id2);
    }
    std::size_t at = 3;
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

stsrl::infra::Tokens encode(const sts::BattleContext& bc, std::span<const sts::search::Action> actions) {
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) throw std::invalid_argument{"nn: Runic Dome hides intents"};
    const auto s = encode_state(bc, CardCoverage::ironclad_events);
    stsrl::infra::Tokens t;
    auto& out = t.data;
    {
        Row r{out, context, 0, 0};
        r.put_all(s.global.numeric); r.put_all(s.global.player);
        r.put(float(s.global.input_state)); r.put(float(s.global.card_selection_task)); r.put(s.global.max_hp / 100.f);
    }
    for (const auto& c : s.cards) { Row r{out, card, c.card_id, 0}; r.put_all(card_features(c)); }
    for (auto m : s.monsters) {
        m.status[13] = m.status[14] = 0;  // opaque simulator counters
        float stances = 0, phase = 0;
        for (const auto& raw : bc.monsters.arr)
            if (raw.id == sts::MonsterId::THE_CHAMP && int(raw.id) == m.monster_id) {
                stances = float(raw.miscInfo & 3); phase = float((raw.miscInfo >> 2) & 1);
            }
        Row r{out, monster, m.monster_id, m.move_id};
        r.put(float(m.previous_move_id)); r.put_all(m.numeric); r.put_all(m.status); r.put(stances); r.put(phase);
    }
    for (const auto& p : s.potions) { Row r{out, potion, p.potion_id, 0}; r.put_all(p.numeric); }
    for (const auto& x : s.relics) { Row r{out, relic, x.relic_id, 0}; r.put_all(x.numeric); }
    for (const auto& move : actions) {
        using T = sts::search::ActionType;
        const auto a = encode_action(bc, move.bits, 0, CardCoverage::ironclad_events);
        const int id = a.source_card ? a.source_card->card_id : a.potion_id.value_or(0);
        Row r{out, action, id, int(a.kind) + 1};
        r.put(float(a.card_selection_task)); r.put(float(a.skips_selection)); r.put(float(a.discards_potion));
        r.put_all(a.interaction.value_or(std::array<float, 6>{}));
        r.put_all(a.source_card ? card_features(*a.source_card) : std::array<float, 17>{});
        std::array<float, 17> subset{};
        float selected = 0;
        if (move.getActionType() == T::MULTI_CARD_SELECT)
            for (int i : move.getSelectedIdxs()) {
                const auto c = encode_action(bc, sts::search::Action{T::SINGLE_CARD_SELECT, i}.bits, 0, CardCoverage::ironclad_events);
                if (!c.source_card) throw std::runtime_error{"nn: subset card missing"};
                const auto f = card_features(*c.source_card);
                for (std::size_t k = 0; k < f.size(); ++k) subset[k] += f[k];
                ++selected;
            }
        r.put(selected); r.put_all(subset);
        r.put_all(a.potion ? a.potion->numeric : decltype(a.potion->numeric){});
    }
    t.count = int(out.size() / width);
    t.actions = int(actions.size());
    return t;
}
}  // namespace stsrl::nn
