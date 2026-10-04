#include "agents/combat/pv/search.hpp"
#include "environments/combat/environment.hpp"
#include "environments/combat/record_v4.hpp"
#include "agents/combat/search/teacher_leaves.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error{message}; }
struct ZeroScore {
    double operator()(double, double, std::int64_t, std::int64_t) const { return 0; }
};
struct LastSelection {
    std::size_t operator()(std::span<const double> scores, std::mt19937_64&) const { return scores.size() - 1; }
};

int main() {
    try {
        std::ifstream file{PV_FIXTURE}; nlohmann::json fight; file >> fight;
        sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
        const auto moves = stsrl::pv::legal_actions(state);
        const auto features = stsrl::pv::encode(state, moves);
        for (std::size_t i = 0; i < features.size(); ++i)
            require(features[i].size() % stsrl::pv::widths[i] == 0, "token width mismatch");
        auto hidden = state;
        hidden.shuffleRng = sts::Random{123}; hidden.aiRng = sts::Random{456};
        hidden.monsters.arr[0].miscInfo |= 0x100;  // unrelated private counter bits
        std::reverse(hidden.cards.drawPile.begin(), hidden.cards.drawPile.end());
        require(stsrl::pv::encode(hidden, moves) == features, "private RNG/draw order leaked");

        auto dome = state;
        dome.player.setHasRelic<sts::RelicId::RUNIC_DOME>(true);
        const auto public_dome = stsrl::pv::encode(dome, moves);
        dome.monsters.arr[0].moveHistory[0] = sts::MonsterMoveId::THE_CHAMP_DEFENSIVE_STANCE;
        dome.monsters.arr[0].miscInfo ^= 7;
        const auto changed_dome = stsrl::pv::encode(dome, moves);
        require(changed_dome == public_dome, "Dome hidden intent/phase leaked");

        // These are checks of our search/objective, with known predictions; no ONNX library test.
        using namespace stsrl::pv;
        Prediction root_prediction{120, std::vector<float>(moves.size(), 0)};
        root_prediction.logits.back() = 5;
        SearchSettings settings;
        settings.batch_size = 8;
        settings.maximum_actions = 8;
        Search<> search{state, stsrl::teacher::public_particles(state, 8), root_prediction, settings};
        int calls = 0;
        auto evaluate = [&](std::span<const Inputs> batch) {
            ++calls;
            std::vector<Prediction> predictions;
            for (const auto& inputs : batch)
                predictions.push_back({120, std::vector<float>(inputs[5].size() / widths[5], 0)});
            return predictions;
        };
        search.run(evaluate, 1);
        require(search.selected_action().bits == moves.back().bits, "root priors did not steer first traversal");
        search.run(evaluate, 31);
        require(search.simulations() == 32 && search.root().visits == 32, "simulation budget/backup mismatch");
        require(search.root().in_flight == 0 && calls > 0, "unfinished search batch");
        double sum = 0;
        for (const auto& edge : search.root().edges) {
            require(edge.in_flight == 0, "unfinished edge reservation");
            sum += edge.value_sum;
        }
        require(sum / search.root().visits > 2, "HP-equivalent values were clipped");
        require(PuctScore{150}(10, 0.5, 3, 1) == 85, "PUCT formula changed");
        // Independent score/selection policies, and eight reserved paths sharing one leaf evaluation.
        Search<ZeroScore, LastSelection> coalesced{state, {state, state}, root_prediction, settings};
        int unique_calls = 0;
        coalesced.run([&](std::span<const Inputs> batch) {
            ++unique_calls;
            require(batch.size() == 1, "pending leaf was evaluated more than once");
            return std::vector<Prediction>{{120, std::vector<float>(batch[0][5].size() / widths[5], 0)}};
        }, 8);
        require(unique_calls == 1 && coalesced.root().edges.back().visits == 8, "pending paths not coalesced");
        // Oracle edges have one deterministic outcome, and a played child retains its visits.
        settings.oracle = true;
        Search<> oracle{state, {state}, root_prediction, settings};
        oracle.run(evaluate, 128);
        std::function<void(const Search<>::Node&)> check_outcomes = [&](const auto& node) {
            for (const auto& edge : node.edges) {
                require(edge.outcomes.size() <= 1, "oracle edge has multiple outcomes");
                for (const auto& [key, child] : edge.outcomes) check_outcomes(*child);
            }
        };
        check_outcomes(oracle.root());
        auto played = state;
        const auto action = oracle.selected_action();
        played = oracle.oracle_state();
        action.execute(played);
        auto wrong_hidden = played;
        wrong_hidden.shuffleRng = sts::Random{123};
        bool hidden_rejected = false;
        try { oracle.advance(action, wrong_hidden); } catch (const std::logic_error&) { hidden_rejected = true; }
        require(hidden_rejected, "reuse accepted wrong hidden state");
        require(oracle.advance(action, played), "searched oracle child not reused");
        require(observation_key(oracle.oracle_state()) == observation_key(played), "reused observation differs");
        require(encode(oracle.oracle_state(), legal_actions(played)) == encode(played, legal_actions(played)), "reused features differ");
        require(oracle.oracle_state().shuffleRng.seed0 == played.shuffleRng.seed0 &&
                oracle.oracle_state().shuffleRng.seed1 == played.shuffleRng.seed1, "reused RNG differs");
        const auto inherited = oracle.root().visits;
        require(inherited > 0, "oracle subtree visits lost");
        oracle.run(evaluate, 32);
        require(oracle.simulations() == 32 && oracle.root().visits == inherited + 32, "reuse budget is not additional sims");
        check_outcomes(oracle.root());
        auto terminal = state;
        terminal.outcome = sts::Outcome::PLAYER_VICTORY;
        terminal.player.curHp = 100;
        terminal.potionCount = 2;
        require(CombatObjective::terminal_value(terminal) == 100, "objective units changed");
        terminal.player.maxHp += 50;
        require(CombatObjective::terminal_value(terminal) == 100, "objective depends on max-HP normalization");
        auto invalid = root_prediction;
        invalid.value = -1;
        bool bad_value_rejected = false;
        try { Search<> bad{state, {state}, invalid}; } catch (const std::invalid_argument&) { bad_value_rejected = true; }
        require(bad_value_rejected, "invalid root value accepted");
        invalid = root_prediction;
        invalid.logits.clear();
        bool bad_policy_rejected = false;
        try { Search<> bad{state, {state}, invalid}; } catch (const std::invalid_argument&) { bad_policy_rejected = true; }
        require(bad_policy_rejected, "invalid root policy accepted");

        // Default encoding coverage remains unchanged; only PV opts into later-run event cards.
        state.cards.hand[0] = sts::CardInstance{sts::CardId::APPARITION};
        bool rejected = false;
        try { (void)stsrl::encode_state(state); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "legacy card capability boundary changed");
        (void)stsrl::pv::encode(state, {});

        state.inputState = sts::InputState::CARD_SELECT;
        state.cardSelectInfo.cardSelectTask = sts::CardSelectTask::GAMBLE;
        const std::array<sts::search::Action, 2> subsets{
            sts::search::Action{sts::search::ActionType::MULTI_CARD_SELECT, 1},
            sts::search::Action{sts::search::ActionType::MULTI_CARD_SELECT, 2}};
        const auto selected = stsrl::pv::encode(state, subsets);
        require(selected[5][80] == float(state.cards.hand[0].id), "first subset identity missing");
        require(selected[5][261 + 80] == float(state.cards.hand[1].id), "second subset identity missing");
        std::cout << "PV feature, objective and search invariants passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
