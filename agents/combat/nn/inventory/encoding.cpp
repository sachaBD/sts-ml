#include "agents/combat/nn/inventory/encoding.hpp"
#include "agents/combat/nn/features.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

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

using CardRows = std::map<CardLocation, int>;

void write_context(std::vector<float>& out, const sts::BattleContext& battle, int held_potions) {
    const int incoming = incoming_damage(battle);
    Row row{out, Kind::context, 0, inventory_version};
    row.put_all(std::array{battle.turn / 20.f, battle.cards.cardsInHand / 10.f,
        battle.cards.drawPile.size() / 64.f, battle.cards.discardPile.size() / 64.f,
        battle.cards.exhaustPile.size() / 64.f, incoming / 100.f,
        std::max(0, incoming - battle.player.block) / 100.f});
    row.put(float(battle.inputState));
    row.put(float(battle.inputState == sts::InputState::CARD_SELECT
                  ? battle.cardSelectInfo.cardSelectTask : sts::CardSelectTask::INVALID));
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

CardRows write_cards(std::vector<float>& out, const sts::BattleContext& battle) {
    struct Copies { std::array<int, 3> counts{}; std::vector<CardLocation> locations; };
    std::map<std::pair<int, std::array<float, 16>>, Copies> inventory;
    std::vector<CardLocation> locations;
    const auto cards = encode_cards(battle, &locations);
    CardRows rows;
    for (std::size_t i = 0; i < cards.size(); ++i) {
        const auto& card = cards[i];
        const auto features = card_features(card);
        if (card.zone == CardZone::hand) {
            Row row{out, Kind::card, card.card_id};
            row.put_all(features);
            row.put(0);  // inventory_v1 reserves this field for offered cards.
            rows.emplace(locations[i], row.index);
        } else {
            auto& copies = inventory[{card.card_id, features}];
            ++copies.counts.at(int(card.zone) - int(CardZone::draw));
            copies.locations.push_back(locations[i]);
        }
    }
    for (const auto& [variant, copies] : inventory) {
        Row row{out, Kind::inventory, variant.first};
        row.put_all(variant.second);
        for (const auto count : copies.counts) row.put(count / 10.f);
        for (const auto location : copies.locations) rows.emplace(location, row.index);
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

std::map<int, int> write_monsters(std::vector<float>& out, const sts::BattleContext& battle) {
    std::vector<int> slots;
    const auto monsters = encode_monsters(battle, &slots);
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

std::map<int, int> write_potions(std::vector<float>& out, const std::vector<PotionToken>& potions,
                                const std::vector<int>& slots) {
    std::map<int, int> rows;
    for (std::size_t i = 0; i < potions.size(); ++i) {
        Row row{out, Kind::potion, potions[i].potion_id};
        row.put_all(potions[i].numeric);
        rows.emplace(slots[i], row.index);
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
            const auto item = encode_action(battle, sts::search::Action{sts::search::ActionType::SINGLE_CARD_SELECT, index});
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
        const auto action = encode_action(battle, move);
        const int id = action.source_card ? action.source_card->card_id : action.potion_id.value_or(0);
        Row row{out, Kind::action, id, int(action.kind) + 1};
        row.put(float(action.card_selection_task));
        row.put(float(action.skips_selection));
        row.put(float(action.discards_potion));
        row.put_all(action.interaction.value_or(std::array<float, 6>{}));
        row.put_all(action.source_card ? card_features(*action.source_card) : std::array<float, 16>{});
        write_card_subset(row, battle, move);
        row.put_all(action.potion ? action.potion->numeric : std::array<float, potion_features>{});
        if (action.source_card && action.source_card->zone != CardZone::offered) {
            const int index = action.kind == EncodedActionKind::card ? move.getSourceIdx() : move.getSelectIdx();
            row.link(inventory_source, cards.at({action.source_card->zone, index}));
        } else if (action.potion_id) row.link(inventory_source, potion_rows.at(move.getSourceIdx()));
        if (action.target_monster) row.link(inventory_target, monster_rows.at(move.getTargetIdx()));
    }
}
}  // namespace

infra::Tokens encode_inventory(const sts::BattleContext& battle, std::span<const sts::search::Action> actions) {
    if (battle.player.hasRelic<sts::RelicId::RUNIC_DOME>()) throw std::invalid_argument{"nn: Runic Dome hides intents"};
    std::vector<int> potion_slots;
    const auto potions = encode_potions(battle, &potion_slots);
    infra::Tokens tokens;
    auto& out = tokens.data;
    write_context(out, battle, int(potions.size()));
    write_player(out, battle.player);
    const auto card_rows = write_cards(out, battle);
    const auto monster_rows = write_monsters(out, battle);
    const auto potion_rows = write_potions(out, potions, potion_slots);
    write_relics(out, encode_relics(battle));
    write_actions(out, battle, actions, card_rows, monster_rows, potion_rows);
    tokens.count = int(out.size() / width);
    tokens.actions = int(actions.size());
    tokens.attended = tokens.count;
    return tokens;
}
}  // namespace stsrl::nn
