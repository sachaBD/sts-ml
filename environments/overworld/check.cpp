// Acceptance checks and benchmark for environments/overworld/macro_sim.hpp.
//   macro_sim_check paths|crn|crn_neow|filter|bench [N]
// Fights are skipped with a fixed fake result (always won; -6 monster, -15 elite, -25 boss, -8 event fight).
#include "environments/overworld/macro_sim.hpp"
#include "sim/search/SimpleAgent.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ms = stsrl::macro_sim;
using sts::GameContext;

namespace {

constexpr std::uint64_t kSeed0 = 700000000000ULL;

ms::BattleResult fake(const GameContext& gc) {
    const int dmg = gc.curRoom == sts::Room::BOSS ? 25 : gc.curRoom == sts::Room::ELITE ? 15
                  : gc.curRoom == sts::Room::MONSTER ? 6 : 8;
    return {true, gc.curHp - dmg, 0, 0, {}};
}

GameContext fresh(std::uint64_t seed) { return GameContext{sts::CharacterClass::IRONCLAD, seed, 20}; }

// ------------------------------------------------------------------ paths
void brute(const sts::Map& m, int x, int y, std::array<sts::Room, 15> seq, int first,
           std::map<std::array<sts::Room, 15>, std::set<int>>& out) {
    const auto& n = m.getNode(x, y);
    seq[static_cast<std::size_t>(y)] = n.room;
    if (y == 14) { out[seq].insert(first); return; }
    for (int e = 0; e < n.edgeCount; ++e) brute(m, n.edges[e], y + 1, seq, first, out);
}

bool check_paths_at(const GameContext& gc) {
    std::map<std::array<sts::Room, 15>, std::set<int>> want;
    std::array<sts::Room, 15> none;
    none.fill(sts::Room::NONE);
    if (gc.curMapNodeY < 0) {
        for (const auto& n : gc.map->nodes[0]) if (n.edgeCount > 0) brute(*gc.map, n.x, 0, none, n.x, want);
    } else if (gc.curMapNodeY < 14) {
        const auto& h = gc.map->getNode(gc.curMapNodeX, gc.curMapNodeY);
        for (int e = 0; e < h.edgeCount; ++e) brute(*gc.map, h.edges[e], gc.curMapNodeY + 1, none, h.edges[e], want);
    }
    const auto got = ms::map_state(gc);
    if (got.paths.size() != want.size()) return false;
    for (const auto& p : got.paths) {
        auto it = want.find(p.rooms);
        if (it == want.end() || std::vector<int>(it->second.begin(), it->second.end()) != p.first_xs) return false;
    }
    return true;
}

int paths(int n) {
    std::vector<std::size_t> counts;
    int bad = 0, checked = 0;
    for (int i = 0; i < n; ++i) {
        auto gc = fresh(kSeed0 + i);
        counts.push_back(ms::map_state(gc).paths.size());
        bad += !check_paths_at(gc); ++checked;
        for (int y = 0; y < 14; ++y)
            for (const auto& node : gc.map->nodes[static_cast<std::size_t>(y)]) {
                if (node.edgeCount == 0) continue;
                auto c = gc;
                c.curMapNodeX = node.x;
                c.curMapNodeY = y;
                bad += !check_paths_at(c); ++checked;
            }
    }
    std::sort(counts.begin(), counts.end());
    std::printf("paths: %d maps, %d positions checked vs brute force, %d mismatches\n", n, checked, bad);
    std::printf("paths from start (distinct room sequences): min %zu median %zu max %zu\n", counts.front(),
                counts[counts.size() / 2], counts.back());
    std::cout << ms::map_json(fresh(kSeed0)).dump().substr(0, 400) << "...\n";
    return bad != 0;
}

// ------------------------------------------------------------------ crn
std::string cards(const sts::CardReward& r) {
    std::string s;
    for (int i = 0; i < r.size(); ++i) s += std::string(r[i].getName()) + (r[i].isUpgraded() ? "+" : "") + ",";
    return s;
}

// What the game offered on entering a screen (floor-tagged).
std::string snapshot(const GameContext& gc) {
    std::ostringstream o;
    o << "f" << gc.floorNum << " " << sts::roomStrings[static_cast<int>(gc.curRoom)] << " ";
    const auto& r = gc.info.rewardsContainer;
    switch (gc.screenState) {
        case sts::ScreenState::BATTLE: o << "battle " << sts::monsterEncounterEnumNames[static_cast<int>(gc.info.encounter)]; break;
        case sts::ScreenState::EVENT_SCREEN: o << "event " << sts::eventGameNames[static_cast<int>(gc.curEvent)]; break;
        case sts::ScreenState::REWARDS:
            o << "rewards gold";
            for (int i = 0; i < r.goldRewardCount; ++i) o << " " << r.gold[i];
            o << " cards";
            for (int i = 0; i < r.cardRewardCount; ++i) o << " [" << cards(r.cardRewards[i]) << "]";
            o << " potions";
            for (int i = 0; i < r.potionCount; ++i) o << " " << sts::potionEnumNames[static_cast<int>(r.potions[i])];
            o << " relics";
            for (int i = 0; i < r.relicCount; ++i) o << " " << sts::relicEnumNames[static_cast<int>(r.relics[i])];
            break;
        case sts::ScreenState::SHOP_ROOM:
            o << "shop";
            for (const auto& c : gc.info.shop.cards) o << " " << c.getName();
            for (const auto p : gc.info.shop.potions) o << " " << sts::potionEnumNames[static_cast<int>(p)];
            for (const auto x : gc.info.shop.relics) o << " " << sts::relicEnumNames[static_cast<int>(x)];
            break;
        case sts::ScreenState::BOSS_RELIC_REWARDS:
            o << "boss_relics";
            for (const auto x : gc.info.bossRelics) o << " " << sts::relicEnumNames[static_cast<int>(x)];
            break;
        default: return "";
    }
    return o.str();
}

struct Runner {
    GameContext gc;
    sts::search::SimpleAgent agent;
    std::vector<std::string> log;
    int last_floor = -1;
    sts::ScreenState last_screen = sts::ScreenState::INVALID;

    void note() {
        if (gc.floorNum == last_floor && gc.screenState == last_screen) return;
        last_floor = gc.floorNum;
        last_screen = gc.screenState;
        if (auto s = snapshot(gc); !s.empty()) log.push_back(s);
    }
    // One step; returns false once the act is over.
    bool step(const ms::Options& opt) {
        if (gc.outcome != sts::GameOutcome::UNDECIDED || gc.act != 1) return false;
        ms::prepare(gc, opt);
        note();
        agent.curGameContext = &gc;
        if (gc.screenState == sts::ScreenState::BATTLE) ms::apply_battle_result(gc, fake(gc));
        else agent.stepOutOfCombat(gc);
        return true;
    }
    bool at_card_choice() const {
        const auto& r = gc.info.rewardsContainer;
        return gc.screenState == sts::ScreenState::REWARDS && r.cardRewardCount > 0 && !(r.potionCount > 0 && gc.potionCount < gc.potionCapacity) &&
               r.relicCount == 0 && !r.sapphireKey && !r.emeraldKey && gc.floorNum >= 1;
    }
};

// Per floor: the logged lines; the floors whose lines differ.
std::vector<int> diverging_floors(const std::vector<std::string>& a, const std::vector<std::string>& b,
                                  std::string* first) {
    std::map<int, std::vector<std::string>> fa, fb;
    auto floor = [](const std::string& s) { return std::stoi(s.substr(1)); };
    for (const auto& s : a) fa[floor(s)].push_back(s);
    for (const auto& s : b) fb[floor(s)].push_back(s);
    std::vector<int> out;
    for (int f = 1; f <= 17; ++f)
        if (fa[f] != fb[f]) {
            if (out.empty() && first) {
                *first = "A:";
                for (const auto& s : fa[f]) *first += " | " + s;
                *first += "\n      B:";
                for (const auto& s : fb[f]) *first += " | " + s;
            }
            out.push_back(f);
        }
    return out;
}

int crn(int n, bool on) {
    const ms::Options opt{on, true};
    int diverged = 0;
    for (int i = 0; i < n; ++i) {
        Runner a; a.gc = fresh(kSeed0 + i);
        // advance to the first card choice after floor 1's fight
        while (!a.at_card_choice() && a.step(opt)) {}
        if (!a.at_card_choice()) { std::printf("seed %d: no card choice\n", i); continue; }
        a.note();
        Runner b = a;
        ms::detach_map(b.gc);  // A crosses into act 2 (transitionToAct rewrites *map)
        const auto& r = a.gc.info.rewardsContainer;
        const int last = r.cardRewardCount - 1;
        const auto taken = std::string(r.cardRewards[last][0].getName());
        const int branch_floor = a.gc.floorNum;
        sts::search::GameAction(sts::search::GameAction::RewardsActionType::CARD, last, 0).execute(a.gc);
        b.gc.info.rewardsContainer.removeCardReward(last);
        while (a.step(opt)) {}
        while (b.step(opt)) {}
        std::string first;
        const auto d = diverging_floors(a.log, b.log, &first);
        diverged += !d.empty();
        std::printf("seed %d: take %s vs skip at floor %d: ", i, taken.c_str(), branch_floor);
        if (d.empty()) std::printf("identical (%zu screens)\n", a.log.size());
        else {
            std::printf("diverge at floors");
            for (int f : d) std::printf(" %d", f);
            std::printf("\n      %s\n", first.c_str());
        }
    }
    std::printf("crn=%d: %d/%d seeds diverged\n", on, diverged, n);
    return 0;
}

// Neow branch: option 0 vs option k (different RNG consumption from floor 0 on), same SimpleAgent afterwards.
int crn_neow(int n, bool on) {
    const ms::Options opt{on, true};
    int diverged = 0, pairs = 0, bad_floors = 0;
    for (int i = 0; i < n; ++i)
        for (int k = 1; k < 4; ++k) {
            Runner a; a.gc = fresh(kSeed0 + i);
            Runner b; b.gc = fresh(kSeed0 + i);
            a.note();
            b.note();
            sts::search::GameAction(0).execute(a.gc);
            sts::search::GameAction(k).execute(b.gc);
            while (a.step(opt)) {}
            while (b.step(opt)) {}
            std::string first;
            const auto d = diverging_floors(a.log, b.log, &first);
            ++pairs;
            bad_floors += static_cast<int>(d.size());
            if (d.empty()) continue;
            ++diverged;
            if (i < 3) {
                std::printf("seed %d neow 0 vs %d: diverge at floors", i, k);
                for (int f : d) std::printf(" %d", f);
                std::printf("\n      %s\n", first.c_str());
            }
        }
    std::printf("crn=%d neow branch: %d/%d pairs diverged on some floor >= 1 (floor 0 = Neow itself differs); "
                "%d/%d pair-floors differ\n", on, diverged, pairs, bad_floors, pairs * 17);
    return 0;
}

// ------------------------------------------------------------------ filter
int filter(int n, bool on) {
    const ms::Options opt{false, on};
    std::map<std::string, int> seen;
    auto relic = [&](sts::RelicId x) { if (ms::is_filtered_relic(x)) ++seen[sts::relicEnumNames[static_cast<int>(x)]]; };
    for (int i = 0; i < n; ++i) {
        auto gc = fresh(kSeed0 + i);
        sts::search::SimpleAgent agent;
        agent.curGameContext = &gc;
        while (gc.outcome == sts::GameOutcome::UNDECIDED && gc.act == 1) {
            ms::prepare(gc, opt);
            if (ms::is_filtered_event(gc.curEvent)) ++seen[sts::eventGameNames[static_cast<int>(gc.curEvent)]];
            for (const auto& x : gc.relics.relics) relic(x.id);
            const auto& r = gc.info.rewardsContainer;
            if (gc.screenState == sts::ScreenState::REWARDS) for (int k = 0; k < r.relicCount; ++k) relic(r.relics[k]);
            if (gc.screenState == sts::ScreenState::SHOP_ROOM) for (const auto x : gc.info.shop.relics) relic(x);
            if (gc.screenState == sts::ScreenState::BATTLE) {
                const bool boss = gc.curRoom == sts::Room::BOSS;
                ms::apply_battle_result(gc, fake(gc));
                if (!boss) continue;
                // boss rewards + boss relic screen
                while (gc.screenState == sts::ScreenState::REWARDS) agent.stepOutOfCombat(gc);
                if (gc.screenState == sts::ScreenState::BOSS_RELIC_REWARDS) for (const auto x : gc.info.bossRelics) relic(x);
                break;
            }
            agent.stepOutOfCombat(gc);
        }
    }
    std::printf("filter=%d, %d runs: filtered content seen (occurrences):", on, n);
    for (const auto& [k, v] : seen) std::printf(" %s=%d", k.c_str(), v);
    std::printf("%s\n", seen.empty() ? " none" : "");
    return on && !seen.empty();
}

// ------------------------------------------------------------------ bench
int bench(int n, ms::Options opt) {
    const auto t0 = std::chrono::steady_clock::now();
    int won = 0;
    long floors = 0;
    for (int i = 0; i < n; ++i) {
        auto gc = fresh(kSeed0 + i);
        won += ms::run_act1(gc, opt, fake);
        floors += gc.floorNum;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("bench crn=%d filter=%d: %d runs in %.3fs = %.0f runs/s (%.1f us/run), boss beaten %d, mean floor %.1f\n",
                opt.crn, opt.filter, n, s, n / s, 1e6 * s / n, won, double(floors) / n);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = argc > 1 ? argv[1] : "";
    const int n = argc > 2 ? std::stoi(argv[2]) : 0;
    if (mode == "paths") return paths(n ? n : 100);
    if (mode == "crn") { crn(n ? n : 20, false); return crn(n ? n : 20, true); }
    if (mode == "crn_neow") { crn_neow(n ? n : 20, false); return crn_neow(n ? n : 20, true); }
    if (mode == "filter") { filter(n ? n : 10000, false); return filter(n ? n : 10000, true); }
    if (mode == "bench") {
        bench(n ? n : 10000, {false, false});
        return bench(n ? n : 10000, {true, true});
    }
    std::cerr << "usage: macro_sim_check paths|crn|crn_neow|filter|bench [N]\n";
    return 2;
}
