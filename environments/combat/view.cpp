#include "environments/combat/view.hpp"
#include "combat/CardSelectInfo.h"
#include "constants/Cards.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Potions.h"
#include "sim/search/Action.h"
#include <algorithm>
#include <sstream>
#include <string>

namespace stsrl::combat {
namespace {
using Json = nlohmann::json;
namespace S = sts;
using sts::search::Action;
using sts::search::ActionType;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

Json card_json(const S::CardInstance& c) {
    return {{"id", static_cast<int>(c.id)}, {"key", lower(S::cardEnumStrings[static_cast<int>(c.id)])},
            {"name", S::getCardName(c.id)}, {"type", lower(S::cardTypeStrings[static_cast<int>(S::getCardType(c.id))])},
            {"upgraded", c.getUpgradeCount()}, {"cost", c.cost}, {"cost_for_turn", c.costForTurn}};
}

template <typename It>
Json cards_json(It first, It last, bool sorted) {
    Json out = Json::array();
    for (; first != last; ++first) out.push_back(card_json(*first));
    if (sorted)
        std::sort(out.begin(), out.end(), [](const Json& a, const Json& b) {
            return std::pair{a["key"].get<std::string>(), a["upgraded"].get<int>()} <
                   std::pair{b["key"].get<std::string>(), b["upgraded"].get<int>()};
        });
    return out;
}

Json player_statuses(const S::Player& p) {
    Json out = Json::object();
    for (int s = 1; s < static_cast<int>(::PlayerStatus::THE_BOMB); ++s) {
        const auto ps = static_cast<::PlayerStatus>(s);
        if (!p.hasStatusRuntime(ps)) continue;
        const bool core = ps == ::PlayerStatus::ARTIFACT || ps == ::PlayerStatus::DEXTERITY ||
                          ps == ::PlayerStatus::STRENGTH || ps == ::PlayerStatus::FOCUS;
        const int amount = core || p.statusMap.count(ps) ? p.getStatusRuntime(ps) : 1;
        if (amount != 0) out[lower(playerStatusEnumStrings[s])] = amount;
    }
    return out;
}

Json monster_json(const S::BattleContext& bc, const S::Monster& m) {
    Json out{{"id", lower(S::monsterIdStrings[static_cast<int>(m.id)])}, {"hp", m.curHp}, {"max_hp", m.maxHp},
             {"block", m.block}, {"alive", m.isAlive()}};
    const auto move = [](S::MMID id) { return id == S::MMID::INVALID ? Json() : Json(S::monsterMoveStrings[static_cast<int>(id)]); };
    out["intent"] = move(m.moveHistory[0]);
    out["last_move"] = move(m.moveHistory[1]);
    if (m.moveHistory[0] != S::MMID::INVALID && m.isAttacking()) {
        const auto base = m.getMoveBaseDamage(bc);
        out["intent_damage"] = m.calculateDamageToPlayer(bc, base.damage);
        out["intent_hits"] = base.attackCount;
    }
    Json statuses = Json::object();
    for (int s = 0; s < static_cast<int>(S::MS::INVALID); ++s) {
        const auto ms = static_cast<S::MonsterStatus>(s);
        const int amount = m.getStatusInternal(ms);
        if (m.hasStatusInternal(ms) && amount != 0) statuses[lower(S::monsterStatusEnumStrings[s])] = amount;
    }
    if (m.strength) statuses["strength"] = m.strength;
    out["statuses"] = statuses;
    return out;
}

std::string kind(const S::BattleContext& bc) {
    if (bc.outcome != S::Outcome::UNDECIDED) return "done";
    if (bc.inputState == S::InputState::PLAYER_NORMAL) return "play";
    if (bc.inputState == S::InputState::CARD_SELECT)
        return lower(S::cardSelectTaskStrings[static_cast<int>(bc.cardSelectInfo.cardSelectTask)]);
    return "unsupported";
}

Json legal_json(const S::BattleContext& bc) {
    Json legal = Json::array();
    if (bc.outcome != S::Outcome::UNDECIDED) return legal;
    for (const auto& a : Action::getAllActionsInState(bc)) {
        std::ostringstream desc;
        a.printDesc(desc, bc);
        Json row{{"bits", a.bits}, {"desc", desc.str()}};
        switch (a.getActionType()) {
            case ActionType::CARD: row["type"] = "card"; row["source"] = a.getSourceIdx(); row["target"] = a.getTargetIdx(); break;
            case ActionType::POTION: row["type"] = "potion"; row["source"] = a.getSourceIdx(); row["target"] = a.getTargetIdx(); break;
            case ActionType::END_TURN: row["type"] = "end_turn"; break;
            case ActionType::SINGLE_CARD_SELECT: row["type"] = "select"; row["source"] = a.getSelectIdx(); break;
            default: row["type"] = "other";
        }
        legal.push_back(std::move(row));
    }
    return legal;
}
}  // namespace

Json view(const S::BattleContext& bc) {
    Json v;
    v["kind"] = kind(bc);
    v["won"] = bc.outcome == S::Outcome::PLAYER_VICTORY;
    v["turn"] = bc.turn;
    const auto& p = bc.player;
    v["player"] = {{"hp", p.curHp}, {"max_hp", p.maxHp}, {"block", p.block}, {"energy", p.energy},
                   {"energy_per_turn", p.energyPerTurn}, {"statuses", player_statuses(p)}};
    v["monsters"] = Json::array();
    for (int i = 0; i < bc.monsters.monsterCount; ++i) v["monsters"].push_back(monster_json(bc, bc.monsters.arr[i]));
    const auto& cards = bc.cards;
    v["hand"] = cards_json(cards.hand.begin(), cards.hand.begin() + cards.cardsInHand, false);
    v["draw"] = cards_json(cards.drawPile.begin(), cards.drawPile.end(), true);
    v["discard"] = cards_json(cards.discardPile.begin(), cards.discardPile.end(), false);
    v["exhaust"] = cards_json(cards.exhaustPile.begin(), cards.exhaustPile.end(), false);
    Json potions = Json::array();
    for (int i = 0; i < bc.potionCapacity; ++i)
        potions.push_back(bc.potions[i] == S::Potion::EMPTY_POTION_SLOT || bc.potions[i] == S::Potion::INVALID
                              ? Json() : Json(S::potionNames[static_cast<int>(bc.potions[i])]));
    v["potions"] = potions;
    v["legal"] = legal_json(bc);
    return v;
}
}  // namespace stsrl::combat
