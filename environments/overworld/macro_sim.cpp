#include "environments/overworld/macro_sim.hpp"

#include "environments/overworld/game_state.hpp"
#include "environments/overworld/observation.hpp"
#include "sim/search/SimpleAgent.h"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>

namespace stsrl::macro_sim {
namespace {

constexpr int kRows = 15;

using Seq = std::array<sts::Room, kRows>;

// Distinct room sequences (rows y..14) from node (x, y); memoised per node.
using SuffixSet = std::vector<Seq>;

const SuffixSet& suffixes(const sts::Map& map, int x, int y, std::map<std::pair<int, int>, SuffixSet>& memo) {
    const auto key = std::make_pair(x, y);
    if (auto it = memo.find(key); it != memo.end()) return it->second;
    const auto& node = map.getNode(x, y);
    SuffixSet out;
    if (y == kRows - 1 || node.edgeCount == 0) {
        Seq s;
        s.fill(sts::Room::NONE);
        s[static_cast<std::size_t>(y)] = node.room;
        out.push_back(s);
    } else {
        for (int e = 0; e < node.edgeCount; ++e)
            for (auto s : suffixes(map, node.edges[e], y + 1, memo)) {
                s[static_cast<std::size_t>(y)] = node.room;
                out.push_back(s);
            }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    return memo.emplace(key, std::move(out)).first->second;
}

std::uint64_t splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

}  // namespace

MapState map_state(const sts::GameContext& gc) {
    MapState st;
    st.y = gc.curMapNodeY;
    st.x = gc.curMapNodeX;
    if (!gc.map || st.y >= kRows - 1) return st;
    std::vector<int> next;
    if (st.y < 0) {
        for (const auto& node : gc.map->nodes[0])
            if (node.edgeCount > 0) next.push_back(node.x);
    } else {
        const auto& here = gc.map->getNode(st.x, st.y);
        for (int e = 0; e < here.edgeCount; ++e) next.push_back(here.edges[e]);
    }
    std::sort(next.begin(), next.end());
    next.erase(std::unique(next.begin(), next.end()), next.end());
    std::map<std::pair<int, int>, SuffixSet> memo;
    std::map<Seq, std::vector<int>> merged;
    for (const int x : next)
        for (const auto& s : suffixes(*gc.map, x, st.y + 1, memo)) merged[s].push_back(x);
    for (auto& [rooms, xs] : merged) st.paths.push_back({rooms, std::move(xs)});
    return st;
}

nlohmann::json map_json(const sts::GameContext& gc) {
    using Json = nlohmann::json;
    Json nodes = Json::array();
    if (gc.map)
        for (int y = 0; y < kRows; ++y)
            for (const auto& node : gc.map->nodes[static_cast<std::size_t>(y)]) {
                if (node.room == sts::Room::NONE) continue;
                Json edges = Json::array();
                for (int e = 0; e < node.edgeCount; ++e) edges.push_back(node.edges[e]);
                nodes.push_back({{"x", node.x}, {"y", node.y}, {"room", sts::roomStrings[static_cast<int>(node.room)]},
                                 {"edges", edges}});
            }
    const auto st = map_state(gc);
    Json paths = Json::array();
    for (const auto& p : st.paths) {
        std::string rooms;
        for (const auto r : p.rooms) rooms.push_back(sts::getRoomSymbol(r));
        paths.push_back({{"rooms", rooms}, {"first_xs", p.first_xs}});
    }
    Json burning = nullptr;
    if (gc.map && gc.map->burningEliteX >= 0) burning = {{"x", gc.map->burningEliteX}, {"y", gc.map->burningEliteY}};
    return {{"current", {{"y", st.y}, {"x", st.x}, {"floor", gc.floorNum}}},
            {"burning_elite", burning},
            {"nodes", nodes},
            {"paths", paths}};
}

nlohmann::json state_json(const sts::GameContext& gc) {
    auto j = game_state::state(gc);
    j["act"] = gc.act;
    j["boss"] = game_state::lower(sts::monsterEncounterEnumNames[static_cast<int>(gc.boss)]);
    j["map"] = map_json(gc);
    j["overworld"] = stsrl::overworld::observation_json(gc);
    return j;
}

void detach_map(sts::GameContext& gc) {
    if (gc.map) gc.map = std::make_shared<sts::Map>(*gc.map);
}

void crn_reseed(sts::GameContext& gc) {
    if (gc.screenState != sts::ScreenState::MAP_SCREEN) return;
    const auto floor = static_cast<std::uint64_t>(gc.floorNum + 1);
    const auto base = splitmix(gc.seed ^ splitmix(floor * 0x100 + 0xC7));
    int stream = 0;
    for (auto* rng : {&gc.cardRng, &gc.potionRng, &gc.relicRng, &gc.eventRng, &gc.merchantRng, &gc.treasureRng,
                      &gc.monsterRng})
        *rng = sts::Random(splitmix(base + static_cast<std::uint64_t>(++stream)));
}

void apply_battle_result(sts::GameContext& gc, const BattleResult& r) {
    if (gc.screenState != sts::ScreenState::BATTLE)
        throw std::logic_error{"apply_battle_result: not on a BATTLE screen"};
    for (const int slot : r.potions_used) {
        if (slot < 0 || slot >= gc.potionCapacity) throw std::out_of_range{"apply_battle_result: potion slot"};
        auto& p = gc.potions[static_cast<std::size_t>(slot)];
        if (p != sts::Potion::EMPTY_POTION_SLOT && p != sts::Potion::INVALID) {
            p = sts::Potion::EMPTY_POTION_SLOT;
            --gc.potionCount;
        }
    }
    gc.maxHp = std::max(1, gc.maxHp + r.max_hp_delta);
    gc.gold = std::max(0, gc.gold + r.gold_delta);
    gc.info.stolenGold = 0;
    if (!r.won) {
        gc.curHp = 0;
        gc.outcome = sts::GameOutcome::PLAYER_LOSS;
        return;
    }
    gc.curHp = std::clamp(r.hp, 1, gc.maxHp);
    for (const auto& relic : gc.relics.relics) {  // BattleContext::updateRelicsOnExit, victory heals
        switch (relic.id) {
            case sts::RelicId::BURNING_BLOOD: gc.playerHeal(6); break;
            case sts::RelicId::BLACK_BLOOD: gc.playerHeal(12); break;
            case sts::RelicId::MEAT_ON_THE_BONE:
                if (gc.curHp <= gc.maxHp / 2) gc.playerHeal(12);
                break;
            default: break;
        }
    }
    gc.regainControl();
}

bool is_filtered_relic(sts::RelicId relic) {
    using R = sts::RelicId;
    return relic == R::PRAYER_WHEEL || relic == R::LIZARD_TAIL || relic == R::PRISMATIC_SHARD ||
           relic == R::NECRONOMICON;
}

bool is_filtered_event(sts::Event event) {
    using E = sts::Event;
    return event == E::DEAD_ADVENTURER || event == E::WE_MEET_AGAIN || event == E::FALLING ||
           event == E::CURSED_TOME;
}

void filter_content(sts::GameContext& gc) {
    for (auto* pool : {&gc.commonRelicPool, &gc.uncommonRelicPool, &gc.rareRelicPool, &gc.shopRelicPool,
                       &gc.bossRelicPool})
        std::erase_if(*pool, is_filtered_relic);
    for (auto* list : {&gc.eventList, &gc.shrineList, &gc.specialOneTimeEventList})
        std::erase_if(*list, is_filtered_event);
}

void prepare(sts::GameContext& gc, const Options& options) {
    if (options.filter) filter_content(gc);
    if (options.crn) crn_reseed(gc);
}

bool run_act1(sts::GameContext& gc, const Options& options, const FightFn& fight) {
    sts::search::SimpleAgent agent;
    agent.curGameContext = &gc;
    while (gc.outcome == sts::GameOutcome::UNDECIDED && gc.act == 1) {
        prepare(gc, options);
        if (gc.screenState == sts::ScreenState::BATTLE) {
            const bool boss = gc.curRoom == sts::Room::BOSS;
            apply_battle_result(gc, fight(gc));
            if (boss && gc.outcome == sts::GameOutcome::UNDECIDED) return true;
            continue;
        }
        agent.stepOutOfCombat(gc);
    }
    return gc.outcome != sts::GameOutcome::PLAYER_LOSS && gc.act > 1;
}

}  // namespace stsrl::macro_sim
