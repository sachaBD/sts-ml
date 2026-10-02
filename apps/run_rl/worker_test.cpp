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
    ctx.max_act = 4;
    gc.transitionToAct(3);
    gc.curRoom = sts::Room::BOSS;
    gc.info.encounter = gc.secondBoss;
    Player heart_run{gc, {}, ctx, nullptr};
    heart_run.acts_cleared = 2;
    heart_run.fight();
    require(!heart_run.done() && heart_run.acts_cleared == 3, "Act 3 clear is not a Heart clear");
    gc.transitionToAct(4);
    gc.curRoom = sts::Room::BOSS;
    gc.info.encounter = sts::MonsterEncounter::THE_HEART;
    heart_run.fight();
    require(heart_run.done() && heart_run.boss_beaten && heart_run.acts_cleared == 4, "Heart is fourth-act clear");
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
void heart_key_options() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 20};
    gc.screenState = sts::ScreenState::REST_ROOM;
    gc.regainControlAction = [](GameContext& g) { g.screenState = sts::ScreenState::MAP_SCREEN; };
    std::vector<std::string> keys;
    const auto options = rest_options(gc, keys, true);
    bool recall = false;
    for (const auto& o : options) if (o.label["action"] == "recall") {
        recall = true;
        require(o.after["overworld"]["keys"]["ruby"] == true, "recall after-state includes real ruby key");
    }
    require(recall, "Heart rest options offer recall");
    require(!gc.redKey, "building key after-states must not change live run");
    auto ctx = context(); ctx.max_act = 4;
    gc.screenState = sts::ScreenState::REWARDS;
    gc.info.rewardsContainer.sapphireKey = true;
    gc.info.rewardsContainer.relicCount = 1;
    gc.info.rewardsContainer.relics[0] = sts::RelicId::VAJRA;
    Player p{gc, {}, ctx, nullptr};
    p.step();
    require(gc.blueKey, "Heart fallback takes sapphire before competing relic");
    require(gc.info.rewardsContainer.relicCount == 0, "sapphire really consumes chest relic");
    require(!gc.hasRelic(sts::RelicId::VAJRA), "sapphire is not a free key plus relic");
    require(slim_state(gc)["map"].contains("nodes"), "sampled ends retain graph");
}
void heart_transition_smoke() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 20};
    gc.transitionToAct(3);
    gc.enterAct3VictoryRoom();
    require(gc.outcome == sts::GameOutcome::PLAYER_VICTORY && gc.act == 3, "without keys victory must not enter Act 4");
    gc.outcome = sts::GameOutcome::UNDECIDED;
    gc.obtainKey(sts::Key::RUBY_KEY); gc.obtainKey(sts::Key::SAPPHIRE_KEY); gc.obtainKey(sts::Key::EMERALD_KEY);
    gc.enterAct3VictoryRoom();
    require(gc.act == 4, "real keys unlock Act 4");
    auto ctx = context(); ctx.max_act = 4;
    ctx.fight = [](GameContext& g, bool, int) {
        const auto encounter = encounter_name(g.info.encounter);
        const auto cat = category(g, g.info.encounter);
        const auto hp = g.curHp;
        stsrl::macro_sim::apply_battle_result(g, {true, hp});
        return FightRecord{encounter, cat, hp, true, "heart-smoke"};
    };
    Player p{gc, {}, ctx, nullptr}; p.acts_cleared = 3;
    int ticks = 0;
    while (!p.done() && ++ticks < 100) {
        if (gc.screenState == sts::ScreenState::BATTLE) p.fight();
        else p.simple_step(); // no neural protocol required for injected-combat smoke
    }
    require(p.done() && p.boss_beaten && p.acts_cleared == 4 && p.fights == 2, "Act 4 route completes Spear/Shield then Heart");
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
void rest_lookahead_protocol() {
    GameContext gc{sts::CharacterClass::IRONCLAD, 42, 0};
    gc.curMapNodeY = 14;
    gc.curMapNodeX = 0;
    gc.floorNum = 15;
    gc.curHp = 20;
    gc.curRoom = sts::Room::REST;
    gc.screenState = sts::ScreenState::REST_ROOM;
    gc.regainControlAction = [](GameContext& g) { g.screenState = sts::ScreenState::MAP_SCREEN; };
    const auto live_map = stsrl::macro_sim::map_json(gc);
    auto ctx = context();
    ctx.decide = {"rest", "boss_relic"};
    ctx.rest_samples = 2;
    int sample_fights = 0;
    ctx.rest_fight = [&sample_fights](GameContext& g, bool sample, int) {
        require(sample, "scaled fight must run only in samples");
        ++sample_fights;
        const int hp = g.curHp;
        stsrl::macro_sim::apply_battle_result(g, {true, 30, 0, 0, {}});
        return FightRecord{"test", "boss", hp, true, "test"};
    };
    std::vector<std::string> keys;
    const auto options = rest_options(gc, keys);
    Json values = Json::array();
    for (std::size_t i = 0; i < options.size(); ++i) values.push_back(i == 1 ? 0.9 : 0.1);
    std::string replies = Json{{"choice", 0}, {"values", values}}.dump() + "\n";
    for (int i = 0; i < 50; ++i) replies += "{\"choice\":0,\"values\":[0.6,0.4]}\n";
    Json steps = Json::array();
    Player p{gc, {}, ctx, &steps};
    Protocol protocol{replies};
    require(p.rest_or_path(true), "rest decision handled");
    std::istringstream messages{protocol.output.str()};
    std::string line;
    Json evaluate;
    int pending = 0, sample_boss_choices = 0;
    while (std::getline(messages, line)) {
        const auto m = Json::parse(line);
        require(m["seed"] == 42, "rest samples retain real run seed");
        pending += m.value("lookahead_pending", false);
        if (m.value("decision", std::string{}) == "boss_relic") {
            require(m["lookahead"] == true, "sample boss relic asks must be greedy");
            ++sample_boss_choices;
        }
        if (m.value("decision", std::string{}) == "rest_lookahead") evaluate = m;
    }
    require(pending == 1 && sample_fights == 4, "two options times two samples; no recursive refinement");
    require(sample_boss_choices == 4, "samples resolve boss rewards/relic and cross act");
    require(evaluate["options"].size() == 2 && evaluate["ends"].size() == 2, "rest evaluate pair shape");
    require(evaluate["simple"] == 0 && evaluate["lookahead"] == false, "initial rest maps to pair zero");
    for (const auto& per : evaluate["ends"]) {
        require(per.size() == 2, "sample count");
        for (const auto& end : per)
            require(end["state"]["act"] == 2 && end["state"]["map"]["current"]["y"] == -1,
                    "stop on first next-act map, before another fight");
    }
    require(gc.act == 1 && live_map == stsrl::macro_sim::map_json(gc), "rest branches leave live map unchanged");
    require(steps[0]["lookahead"]["options"] == Json::array({0, 1}) && steps[0]["choice"] == 0,
            "log executed original option and sampled values");
    Json trace;
    require(p.refine_rest(options, 0, 1, Json{}, trace) == 1 && trace.is_null(),
            "null values preserve exploration/simple choice without refinement");
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
    heart_key_options();
    heart_transition_smoke();
    boss_after_states();
    obtain_and_sample_seed();
    rest_lookahead_protocol();
    std::cout << "worker protocol tests passed\n";
}
