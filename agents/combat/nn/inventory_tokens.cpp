#include "agents/combat/nn/inventory_tokens.hpp"
#include "environments/combat/environment.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace stsrl::nn {
namespace {
using Kind = InventoryKind;

float symlog(float x) { return std::copysign(std::log1p(std::abs(x)), x); }

struct Row {
    std::vector<float>& data;
    int index;
    int next = 3;
    Row(std::vector<float>& out, Kind kind, int id = 0, int id2 = 0)
        : data(out), index(int(out.size() / width)) {
        data.resize(data.size() + width, 0);
        data[index * width] = float(kind);
        data[index * width + 1] = float(id);
        data[index * width + 2] = float(id2);
    }
    void put(float value) {
        if (next >= inventory_owner) throw std::logic_error{"inventory_v1: feature overflow"};
        data[index * width + next++] = value;
    }
    template<class R> void put_all(const R& values) { for (auto x : values) put(float(x)); }
    void link(int column, int row) { data[index * width + column] = float(row + 1); }
};

std::array<float, 16> card_features(const CardToken& card) {
    std::array<float, 16> features{float(card.card_type), float(card.target_type)};
    std::copy(card.numeric.begin(), card.numeric.end(), features.begin() + 2);
    return features;
}

using CardKey = std::tuple<int, CardType, TargetType, std::array<float, 14>>;
CardKey key(CardToken card) {
    card.numeric.back() = 0;  // playability is contextual, not an instance modification
    return {card.card_id, card.card_type, card.target_type, card.numeric};
}

int player_amount(const sts::Player& player, PlayerStatus status) {
    if (!player.hasStatusRuntime(status)) return 0;
    if (status == PS::STRENGTH || status == PS::DEXTERITY || status == PS::FOCUS || status == PS::ARTIFACT)
        return player.getStatusRuntime(status);
    const auto found = player.statusMap.find(status);
    return found == player.statusMap.end() ? 1 : found->second;
}
}  // namespace

infra::Tokens encode_inventory(const sts::BattleContext& bc, std::span<const sts::search::Action> actions) {
    if (bc.player.hasRelic<sts::RelicId::RUNIC_DOME>()) throw std::invalid_argument{"nn: Runic Dome hides intents"};
    std::vector<int> monster_slots;
    const auto state = encode_state(bc, CardCoverage::ironclad_events, &monster_slots);
    infra::Tokens tokens;
    auto& out = tokens.data;
    const auto& player = bc.player;
    {
        Row row{out, Kind::context, 0, inventory_version};
        row.put_all(std::array{state.global.numeric[0], state.global.numeric[12], state.global.numeric[13],
                              state.global.numeric[14], state.global.numeric[15], state.global.numeric[16],
                              state.global.numeric[17]});
        row.put(float(state.global.input_state));
        row.put(float(state.global.card_selection_task));
        row.put_all(std::array{state.global.player[9], state.global.player[10]});
    }
    const int player_row = int(out.size() / width);
    {
        Row row{out, Kind::player};
        row.put_all(std::array{player.curHp / 100.f, player.maxHp / 100.f,
            player.maxHp ? player.curHp / float(player.maxHp) : 0.f, player.block / 100.f,
            player.energy / 10.f, player.energyPerTurn / 10.f, player.cardDrawPerTurn / 10.f,
            player.cardsPlayedThisTurn / 20.f, player.attacksPlayedThisTurn / 20.f,
            player.skillsPlayedThisTurn / 20.f, player.cardsDiscardedThisTurn / 20.f});
    }
    for (int id = 1; id < int(PS::THE_BOMB); ++id) {
        const auto status = static_cast<PlayerStatus>(id);
        const int amount = player_amount(player, status);
        if (!amount) continue;
        Row row{out, Kind::player_effect, id};
        row.put(symlog(float(amount)));
        row.put(float(id < 32 && (player.justAppliedBits & (1U << id))));
        row.put(status == PS::COMBUST ? symlog(player.combustHpLoss) : 0.f);
        row.put(status == PS::PANACHE ? symlog(player.panacheCounter) : 0.f);
        row.link(inventory_owner, player_row);
    }
    if (player.bomb1 || player.bomb2 || player.bomb3) {
        Row row{out, Kind::player_effect, int(PS::THE_BOMB)};
        row.put_all(std::array{symlog(player.bomb1), symlog(player.bomb2), symlog(player.bomb3)});
        row.link(inventory_owner, player_row);
    }

    std::map<CardKey, int> hand_rows;
    struct Inventory { CardToken card; std::array<int, 3> counts{}; int row = -1; };
    std::map<CardKey, Inventory> inventory;
    for (const auto& card : state.cards) {
        if (card.zone == CardZone::hand || card.zone == CardZone::offered) {
            Row row{out, Kind::card, card.card_id};
            row.put_all(card_features(card));
            row.put(float(card.zone == CardZone::offered));
            hand_rows.try_emplace(key(card), row.index);
        } else {
            auto [it, inserted] = inventory.try_emplace(key(card), Inventory{card});
            ++it->second.counts.at(int(card.zone) - int(CardZone::draw));
        }
    }
    for (auto& [_, entry] : inventory) {
        Row row{out, Kind::inventory, entry.card.card_id};
        entry.row = row.index;
        row.put_all(card_features(entry.card));
        for (const auto count : entry.counts) row.put(count / 10.f);
    }

    std::map<int, int> monster_rows;
    for (std::size_t i = 0; i < state.monsters.size(); ++i) {
        const auto& monster = state.monsters[i];
        const auto& raw = bc.monsters.arr[monster_slots[i]];
        Row row{out, Kind::monster, monster.monster_id, monster.move_id};
        monster_rows.emplace(monster_slots[i], row.index);
        row.put(float(monster.previous_move_id));
        // Strength, weak and vulnerable live in their own effect tokens.
        auto numeric = monster.numeric;
        numeric[5] = numeric[6] = numeric[7] = 0;
        row.put_all(numeric);
        row.put(float(raw.doesEscapeNext()));
        row.put(raw.id == sts::MonsterId::THE_CHAMP ? float(raw.miscInfo & 3) : 0.f);
        row.put(raw.id == sts::MonsterId::THE_CHAMP ? float((raw.miscInfo >> 2) & 1) : 0.f);
        const int owner = row.index;
        for (int id = 0; id < int(sts::MS::INVALID); ++id) {
            const auto status = static_cast<sts::MS>(id);
            const int amount = raw.getStatusInternal(status);
            if (!amount) continue;
            Row effect{out, Kind::monster_effect, id + 1};
            effect.put(symlog(float(amount)));
            bool applied = false;
            if (status == sts::MS::WEAK) applied = raw.wasJustApplied<sts::MS::WEAK>();
            if (status == sts::MS::VULNERABLE) applied = raw.wasJustApplied<sts::MS::VULNERABLE>();
            effect.put(float(applied));
            effect.link(inventory_owner, owner);
        }
    }
    std::map<int, int> potion_rows;
    for (const auto& potion : state.potions) {
        Row row{out, Kind::potion, potion.potion_id};
        row.put_all(potion.numeric);
        potion_rows.try_emplace(potion.potion_id, row.index);
    }
    for (const auto& relic : state.relics) {
        Row row{out, Kind::relic, relic.relic_id};
        row.put_all(relic.numeric);
    }
    for (const auto move : actions) {
        using ActionType = sts::search::ActionType;
        const auto action = encode_action(bc, move.bits, 0, CardCoverage::ironclad_events);
        Row row{out, Kind::action, action.source_card ? action.source_card->card_id : action.potion_id.value_or(0),
                int(action.kind) + 1};
        row.put(float(action.card_selection_task));
        row.put(float(action.skips_selection));
        row.put(float(action.discards_potion));
        row.put_all(action.interaction.value_or(std::array<float, 6>{}));
        row.put_all(action.source_card ? card_features(*action.source_card) : std::array<float, 16>{});
        std::array<float, 16> subset{};
        int selected = 0;
        if (move.getActionType() == ActionType::MULTI_CARD_SELECT)
            for (const int index : move.getSelectedIdxs()) {
                const auto item = encode_action(bc, sts::search::Action{ActionType::SINGLE_CARD_SELECT, index}.bits,
                                                0, CardCoverage::ironclad_events);
                if (!item.source_card) throw std::logic_error{"inventory_v1: selected card missing"};
                const auto features = card_features(*item.source_card);
                for (std::size_t j = 0; j < features.size(); ++j) subset[j] += features[j];
                ++selected;
            }
        row.put(float(selected));
        row.put_all(subset);
        row.put_all(action.potion ? action.potion->numeric : std::array<float, potion_features>{});
        if (action.source_card) {
            const auto variant = key(*action.source_card);
            if (action.source_card->zone == CardZone::hand) {
                if (const auto found = hand_rows.find(variant); found != hand_rows.end()) row.link(inventory_source, found->second);
            } else if (const auto found = inventory.find(variant); found != inventory.end()) {
                row.link(inventory_source, found->second.row);
            }
        } else if (action.potion_id) {
            row.link(inventory_source, potion_rows.at(*action.potion_id));
        }
        if (action.target_monster) row.link(inventory_target, monster_rows.at(move.getTargetIdx()));
    }
    tokens.count = int(out.size() / width);
    tokens.actions = int(actions.size());
    tokens.attended = tokens.count;
    return tokens;
}
}  // namespace stsrl::nn
