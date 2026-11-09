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

struct CardRows {
    std::map<CardKey, int> hand, inventory;

    int source_row(const CardToken& card) const {
        const auto& rows = card.zone == CardZone::hand ? hand : inventory;
        const auto found = rows.find(key(card));
        return found == rows.end() ? -1 : found->second;
    }
};

void write_context(std::vector<float>& out, const sts::BattleContext& battle, const EncodedCombatState& state) {
    // Reuse the incoming-damage aggregates calculated by encode_state.
    const float incoming = state.global.numeric[16], unblocked = state.global.numeric[17];
    const int held_potions = int(state.potions.size());
    Row row{out, Kind::context, 0, inventory_version};
    row.put_all(std::array{battle.turn / 20.f, battle.cards.cardsInHand / 10.f,
        battle.cards.drawPile.size() / 64.f, battle.cards.discardPile.size() / 64.f,
        battle.cards.exhaustPile.size() / 64.f, incoming, unblocked});
    row.put(float(state.global.input_state));
    row.put(float(state.global.card_selection_task));
    row.put(held_potions / 5.f);
    row.put(std::max(0, battle.potionCapacity - held_potions) / 5.f);
}

int player_amount(const sts::Player& player, PlayerStatus status) {
    if (!player.hasStatusRuntime(status)) return 0;
    if (status == PS::STRENGTH || status == PS::DEXTERITY || status == PS::FOCUS || status == PS::ARTIFACT)
        return player.getStatusRuntime(status);
    const auto found = player.statusMap.find(status);
    return found == player.statusMap.end() ? 1 : found->second;
}

void write_player_effects(std::vector<float>& out, const sts::Player& player, int owner) {
    for (int id = 1; id < int(PS::THE_BOMB); ++id) {
        const auto status = static_cast<PlayerStatus>(id);
        const int amount = player_amount(player, status);
        if (!amount) continue;
        Row row{out, Kind::player_effect, id};
        row.put(symlog(float(amount)));
        row.put(float(id < 32 && (player.justAppliedBits & (1U << id))));
        row.put(status == PS::COMBUST ? symlog(player.combustHpLoss) : 0.f);
        row.put(status == PS::PANACHE ? symlog(player.panacheCounter) : 0.f);
        row.link(inventory_owner, owner);
    }
    if (player.bomb1 || player.bomb2 || player.bomb3) {
        Row row{out, Kind::player_effect, int(PS::THE_BOMB)};
        row.put_all(std::array{symlog(player.bomb1), symlog(player.bomb2), symlog(player.bomb3)});
        row.link(inventory_owner, owner);
    }
}

void write_player(std::vector<float>& out, const sts::Player& player) {
    Row row{out, Kind::player};
    row.put_all(std::array{player.curHp / 100.f, player.maxHp / 100.f,
        player.maxHp ? player.curHp / float(player.maxHp) : 0.f, player.block / 100.f,
        player.energy / 10.f, player.energyPerTurn / 10.f, player.cardDrawPerTurn / 10.f,
        player.cardsPlayedThisTurn / 20.f, player.attacksPlayedThisTurn / 20.f,
        player.skillsPlayedThisTurn / 20.f, player.cardsDiscardedThisTurn / 20.f});
    write_player_effects(out, player, row.index);
}

CardRows write_cards(std::vector<float>& out, const std::vector<CardToken>& cards) {
    struct Group { CardToken card; std::array<int, 3> counts{}; };
    std::map<CardKey, Group> inventory;
    CardRows rows;
    for (const auto& card : cards) {
        if (card.zone == CardZone::hand || card.zone == CardZone::offered) {
            Row row{out, Kind::card, card.card_id};
            row.put_all(card_features(card));
            row.put(float(card.zone == CardZone::offered));
            rows.hand.try_emplace(key(card), row.index);
        } else {
            auto& group = inventory.try_emplace(key(card), Group{card}).first->second;
            ++group.counts.at(int(card.zone) - int(CardZone::draw));
        }
    }
    for (const auto& [variant, group] : inventory) {
        Row row{out, Kind::inventory, group.card.card_id};
        row.put_all(card_features(group.card));
        for (const auto count : group.counts) row.put(count / 10.f);
        rows.inventory.emplace(variant, row.index);
    }
    return rows;
}

void write_monster_effects(std::vector<float>& out, const sts::Monster& monster, int owner) {
    for (int id = 0; id < int(sts::MS::INVALID); ++id) {
        const auto status = static_cast<sts::MS>(id);
        const int amount = monster.getStatusInternal(status);
        if (!amount) continue;
        Row row{out, Kind::monster_effect, id + 1};
        row.put(symlog(float(amount)));
        bool applied = false;
        if (status == sts::MS::WEAK) applied = monster.wasJustApplied<sts::MS::WEAK>();
        if (status == sts::MS::VULNERABLE) applied = monster.wasJustApplied<sts::MS::VULNERABLE>();
        row.put(float(applied));
        row.link(inventory_owner, owner);
    }
}

std::map<int, int> write_monsters(std::vector<float>& out, const sts::BattleContext& battle,
                                 const std::vector<MonsterToken>& monsters, const std::vector<int>& slots) {
    std::map<int, int> rows;
    for (std::size_t i = 0; i < monsters.size(); ++i) {
        const auto& monster = monsters[i];
        const auto& raw = battle.monsters.arr[slots[i]];
        Row row{out, Kind::monster, monster.monster_id, monster.move_id};
        row.put(float(monster.previous_move_id));
        // Strength, weak and vulnerable live in their own effect tokens.
        auto numeric = monster.numeric;
        numeric[5] = numeric[6] = numeric[7] = 0;
        row.put_all(numeric);
        row.put(float(raw.doesEscapeNext()));
        row.put(raw.id == sts::MonsterId::THE_CHAMP ? float(raw.miscInfo & 3) : 0.f);
        row.put(raw.id == sts::MonsterId::THE_CHAMP ? float((raw.miscInfo >> 2) & 1) : 0.f);
        rows.emplace(slots[i], row.index);
        write_monster_effects(out, raw, row.index);
    }
    return rows;
}

std::map<int, int> write_potions(std::vector<float>& out, const std::vector<PotionToken>& potions) {
    std::map<int, int> rows;
    for (const auto& potion : potions) {
        Row row{out, Kind::potion, potion.potion_id};
        row.put_all(potion.numeric);
        rows.try_emplace(potion.potion_id, row.index);
    }
    return rows;
}

void write_relics(std::vector<float>& out, const std::vector<RelicToken>& relics) {
    for (const auto& relic : relics) {
        Row row{out, Kind::relic, relic.relic_id};
        row.put_all(relic.numeric);
    }
}

void write_card_subset(Row& row, const sts::BattleContext& battle, sts::search::Action move) {
    std::array<float, 16> features{};
    int count = 0;
    if (move.getActionType() == sts::search::ActionType::MULTI_CARD_SELECT)
        for (const int index : move.getSelectedIdxs()) {
            const auto item = encode_action(battle, sts::search::Action{sts::search::ActionType::SINGLE_CARD_SELECT, index}.bits,
                                            0, CardCoverage::ironclad_events);
            if (!item.source_card) throw std::logic_error{"inventory_v1: selected card missing"};
            const auto selected = card_features(*item.source_card);
            for (std::size_t j = 0; j < features.size(); ++j) features[j] += selected[j];
            ++count;
        }
    row.put(float(count));
    row.put_all(features);
}

void write_actions(std::vector<float>& out, const sts::BattleContext& battle,
                   std::span<const sts::search::Action> actions, const CardRows& cards,
                   const std::map<int, int>& monster_rows, const std::map<int, int>& potion_rows) {
    for (const auto move : actions) {
        const auto action = encode_action(battle, move.bits, 0, CardCoverage::ironclad_events);
        const int id = action.source_card ? action.source_card->card_id : action.potion_id.value_or(0);
        Row row{out, Kind::action, id, int(action.kind) + 1};
        row.put(float(action.card_selection_task));
        row.put(float(action.skips_selection));
        row.put(float(action.discards_potion));
        row.put_all(action.interaction.value_or(std::array<float, 6>{}));
        row.put_all(action.source_card ? card_features(*action.source_card) : std::array<float, 16>{});
        write_card_subset(row, battle, move);
        row.put_all(action.potion ? action.potion->numeric : std::array<float, potion_features>{});
        if (action.source_card) row.link(inventory_source, cards.source_row(*action.source_card));
        else if (action.potion_id) row.link(inventory_source, potion_rows.at(*action.potion_id));
        if (action.target_monster) row.link(inventory_target, monster_rows.at(move.getTargetIdx()));
    }
}
}  // namespace

infra::Tokens encode_inventory(const sts::BattleContext& battle, std::span<const sts::search::Action> actions) {
    if (battle.player.hasRelic<sts::RelicId::RUNIC_DOME>()) throw std::invalid_argument{"nn: Runic Dome hides intents"};
    std::vector<int> monster_slots;
    const auto state = encode_state(battle, CardCoverage::ironclad_events, &monster_slots);
    infra::Tokens tokens;
    auto& out = tokens.data;
    write_context(out, battle, state);
    write_player(out, battle.player);
    const auto card_rows = write_cards(out, state.cards);
    const auto monster_rows = write_monsters(out, battle, state.monsters, monster_slots);
    const auto potion_rows = write_potions(out, state.potions);
    write_relics(out, state.relics);
    write_actions(out, battle, actions, card_rows, monster_rows, potion_rows);
    tokens.count = int(out.size() / width);
    tokens.actions = int(actions.size());
    tokens.attended = tokens.count;
    return tokens;
}
}  // namespace stsrl::nn
