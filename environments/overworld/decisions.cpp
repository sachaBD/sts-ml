#include "environments/overworld/decisions.hpp"
#include "environments/overworld/game_state.hpp"
#include "environments/overworld/observation.hpp"
#include "sim/search/GameAction.h"
#include <algorithm>
#include <set>

namespace stsrl::overworld {
using Json = nlohmann::json;
using sts::GameContext;
using stsrl::game_state::lower;

std::string outcome_key(const GameContext& gc) {
    std::vector<std::pair<int, int>> deck;
    for (const auto& c : gc.deck.cards) deck.emplace_back(static_cast<int>(c.id), c.getUpgraded());
    std::sort(deck.begin(), deck.end());
    Json k = {{"deck", deck}, {"hp", gc.curHp}, {"max_hp", gc.maxHp},
              {"keys", {gc.redKey, gc.blueKey, gc.greenKey}}};
    for (const auto& r : gc.relics.relics) k["relics"].push_back({static_cast<int>(r.id), r.data});
    return k.dump();
}



std::vector<Option> rest_options(const GameContext& gc, std::vector<std::string>& keys, bool include_recall) {
    std::vector<Option> out;
    std::vector<int> actions{0, 1, 3};
    if (include_recall) actions.push_back(2);
    for (int a : actions) {
        if (!sts::search::GameAction(a).isValidAction(gc)) continue;
        GameContext c = gc;
        sts::search::GameAction(a).execute(c);
        if (a != 1) {
            const auto k = outcome_key(c);
            if (std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
            keys.push_back(k);
            out.push_back({{{"action", a == 0 ? "rest" : a == 2 ? "recall" : "lift"}}, {a}, stsrl::macro_sim::state_json(c)});
            continue;
        }
        if (c.screenState != sts::ScreenState::CARD_SELECT) continue;
        for (int i = 0; i < static_cast<int>(c.info.toSelectCards.size()); ++i) {
            GameContext c2 = c;
            sts::search::GameAction(i).execute(c2);
            const auto k = outcome_key(c2);
            if (std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
            keys.push_back(k);
            const auto& card = c.info.toSelectCards[i].card;
            out.push_back({{{"action", "smith"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(card.id)])},
                            {"upgraded", card.getUpgraded()}}, {1, i}, stsrl::macro_sim::state_json(c2)});
        }
    }
    return out;
}

std::string shop_key(const GameContext& gc) {
    Json k = Json::parse(outcome_key(gc));
    k["gold"] = gc.gold;
    for (int i = 0; i < gc.potionCapacity; ++i) k["potions"].push_back(static_cast<int>(gc.potions[static_cast<std::size_t>(i)]));
    return k.dump();
}

std::vector<Option> shop_options(const GameContext& gc, std::vector<std::string>& keys) {
    using RA = sts::search::GameAction::RewardsActionType;
    const auto& shop = gc.info.shop;
    std::vector<Option> out;
    const auto add = [&](Json label, std::vector<int> actions, const GameContext& after_gc, Json after) {
        const auto k = shop_key(after_gc) + (label.contains("relic") ? label["relic"].dump() : "");
        if (std::find(keys.begin(), keys.end(), k) != keys.end()) return;
        keys.push_back(k);
        out.push_back({std::move(label), std::move(actions), std::move(after)});
    };
    add({{"action", "leave"}}, {static_cast<int>(sts::search::GameAction(RA::SKIP).bits)}, gc, stsrl::macro_sim::state_json(gc));
    for (int i = 0; i < 7; ++i) {
        const sts::search::GameAction a(RA::CARD, i);
        if (!a.isValidAction(gc)) continue;
        GameContext c = gc;
        a.execute(c);
        add({{"action", "card"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(shop.cards[i].getId())])},
             {"upgraded", shop.cards[i].isUpgraded()}, {"price", shop.cardPrice(i)}},
            {static_cast<int>(a.bits)}, c, stsrl::macro_sim::state_json(c));
    }
    for (int i = 0; i < 3; ++i) {
        const sts::search::GameAction a(RA::POTION, i);
        if (!a.isValidAction(gc) || gc.potionCount >= gc.potionCapacity) continue;
        GameContext c = gc;
        a.execute(c);
        add({{"action", "potion"}, {"potion", lower(sts::potionEnumNames[static_cast<int>(shop.potions[i])])},
             {"price", shop.potionPrice(i)}}, {static_cast<int>(a.bits)}, c, stsrl::macro_sim::state_json(c));
    }
    for (int i = 0; i < 3; ++i) {
        const sts::search::GameAction a(RA::RELIC, i);
        if (!a.isValidAction(gc)) continue;
        auto after = stsrl::macro_sim::state_json(gc);
        const auto id = shop.relics[i];
        after["relics"].push_back({{"relic_id", static_cast<int>(id)}, {"data", 0},
                                   {"name", lower(sts::relicEnumNames[static_cast<int>(id)])}});
        after["gold"] = gc.gold - shop.relicPrice(i);
        GameContext keyed = gc;
        keyed.gold -= shop.relicPrice(i);
        keyed.relics.add({id, 0}); // public hypothetical ownership, without peeking at random pickup effects
        after["overworld"] = observation_json(keyed);
        add({{"action", "relic"}, {"relic", lower(sts::relicEnumNames[static_cast<int>(id)])}, {"price", shop.relicPrice(i)}},
            {static_cast<int>(a.bits)}, keyed, std::move(after));
    }
    const sts::search::GameAction remove(RA::CARD_REMOVE);
    if (remove.isValidAction(gc)) {
        GameContext c = gc;
        remove.execute(c);
        if (c.screenState == sts::ScreenState::CARD_SELECT)
            for (int j = 0; j < static_cast<int>(c.info.toSelectCards.size()); ++j) {
                GameContext c2 = c;
                sts::search::GameAction(j).execute(c2);
                const auto& card = c.info.toSelectCards[j].card;
                add({{"action", "remove"}, {"card", lower(sts::cardEnumStrings[static_cast<int>(card.id)])},
                     {"upgraded", card.getUpgraded()}, {"price", shop.removeCost}},
                    {static_cast<int>(remove.bits), static_cast<int>(sts::search::GameAction(j).bits)}, c2,
                    stsrl::macro_sim::state_json(c2));
            }
    }
    return out;
}


}  // namespace stsrl::overworld
