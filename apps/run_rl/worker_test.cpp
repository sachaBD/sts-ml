// Keep protocol tests beside the worker; injected fights avoid expensive combat search.
#define main run_rl_worker_main
#include "apps/run_rl/worker.cpp"
#undef main
#include <sstream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error{message};
}
struct Protocol {
    std::istringstream input;
    std::ostringstream output;
    std::streambuf *in, *out;
    explicit Protocol(std::string replies) : input{std::move(replies)},
        in{std::cin.rdbuf(input.rdbuf())}, out{std::cout.rdbuf(output.rdbuf())} {}
    ~Protocol() { std::cin.rdbuf(in); std::cout.rdbuf(out); }
    Json message() const { return Json::parse(output.str()); }
};
Ctx context() {
    Ctx ctx;
    ctx.seed = 42;
    ctx.max_act = 2;
    ctx.fight = [](GameContext& gc, bool, int) {
        gc.screenState = sts::ScreenState::REWARDS;
        return FightRecord{"test", "boss", gc.curHp, true, "test"};
    };
    return ctx;
}
void stopping_and_guard() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 0};
    auto ctx = context();
    Player p{gc, {}, ctx, nullptr};
    gc.curRoom = sts::Room::BOSS;
    p.fight();
    require(!p.done() && p.acts_cleared == 1, "act 1 boss must not stop act 2 run");
    gc.transitionToAct(2);
    p.fight();
    require(p.done() && p.acts_cleared == 2, "act 2 boss must stop run");
    ctx.max_act = 3;
    gc.ascension = 20;
    gc.transitionToAct(3);
    gc.info.encounter = gc.boss;
    Player double_boss{gc, {}, ctx, nullptr};
    double_boss.acts_cleared = 2;
    double_boss.fight();
    require(!double_boss.done() && double_boss.acts_cleared == 2, "first A20 boss is not act clear");
    gc.info.encounter = gc.secondBoss;
    double_boss.fight();
    require(double_boss.done() && double_boss.acts_cleared == 3, "second A20 boss clears act");
    ctx.max_act = 1;
    gc.act = 1;
    Player q{gc, {}, ctx, nullptr};
    q.fight();
    require(q.done(), "default target must stop at act 1 boss");
    q.ticks = 20000;
    try { q.step(); throw std::runtime_error{"missing guard"}; }
    catch (const std::runtime_error& e) {
        require(std::string{e.what()}.find("screen=") != std::string::npos, "guard must describe screen");
    }
}
void boss_after_states() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 0};
    gc.enterBossTreasureRoom();
    gc.info.bossRelics[0] = sts::RelicId::ASTROLABE;
    gc.info.bossRelics[1] = sts::RelicId::EMPTY_CAGE;
    gc.info.bossRelics[2] = sts::RelicId::CALLING_BELL;
    const auto original_map = stsrl::macro_sim::map_json(gc);
    auto ctx = context();
    Json steps = Json::array();
    Player p{gc, {}, ctx, &steps};
    Protocol protocol{"{\"choice\":0}\n"};
    p.boss_relic();
    const auto m = protocol.message();
    require(m["seed"] == 42 && m["decision"] == "boss_relic", "boss protocol");
    require(m["after"].size() == 4 && m["options"][3]["relic"] == "skip", "boss options");
    require(gc.act == 1 && gc.screenState == sts::ScreenState::CARD_SELECT, "real pickup not auto-resolved");
    require(original_map == stsrl::macro_sim::map_json(gc), "branches must not overwrite live map");
    for (const auto& after : m["after"]) {
        require(after["act"] == 2 && after.contains("boss"), "after-state act/boss metadata");
        require(after["map"]["current"]["y"] == -1, "pickup after-state must reach next act map");
        require(!after["map"]["paths"].empty(), "act 2 after-state paths missing");
    }
    require(steps[0].contains("boss") && steps[0].contains("after"), "boss step logged");
}
void obtain_and_sample_seed() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 0};
    gc.info.toSelectCards.clear();
    gc.info.toSelectCards.push_back(sts::SelectScreenCard{sts::Card{sts::CardId::ANGER}});
    gc.info.toSelectCards.push_back(sts::SelectScreenCard{sts::Card{sts::CardId::FEED}});
    gc.openCardSelectScreen(sts::CardSelectScreenType::OBTAIN, 1, false);
    gc.regainControlAction = [](GameContext& g) { g.screenState = sts::ScreenState::MAP_SCREEN; };
    auto ctx = context();
    ctx.sample = true;
    fresh_randomness(gc, 123);
    Player p{gc, {}, ctx, nullptr};
    const int n = gc.deck.size();
    Protocol protocol{"{\"choice\":1}\n"};
    p.obtain_pick();
    const auto m = protocol.message();
    require(m["seed"] == 42 && m["lookahead"] == true, "sample must report real seed");
    require(m["skip_allowed"] == false && m["options"].size() == 2, "obtain skip legality");
    require(gc.deck.size() == n + 1 && gc.deck.cards.back().id == sts::CardId::FEED, "obtain selected card");
}
}
int main() {
    stopping_and_guard();
    boss_after_states();
    obtain_and_sample_seed();
    std::cout << "worker protocol tests passed\n";
}
