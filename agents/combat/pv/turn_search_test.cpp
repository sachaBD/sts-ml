#include "agents/combat/pv/turn_search.hpp"
#include "agents/combat/pv/turn_state_key.hpp"
#include "environments/combat/record_v4.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace stsrl::pv;
void require(bool value, const char* message) { if (!value) throw std::runtime_error{message}; }
sts::BattleContext replay(sts::BattleContext state, const std::vector<std::uint32_t>& sequence) {
    for (auto bits : sequence) {
        const sts::search::Action action{bits}; require(action.isValidAction(state), "stored action illegal"); action.execute(state);
    }
    return state;
}
int main() {
    try {
        std::ifstream file{PV_FIXTURE}; nlohmann::json fight; file >> fight;
        sts::BattleContext initial; initial.init(stsrl::combat_v4::start_game(fight.at("start")));
        auto state = initial;
        state.player = sts::Player{}; state.player.cc = sts::CharacterClass::IRONCLAD;
        state.player.curHp = state.player.maxHp = 100; state.player.energy = 0;
        state.cards = sts::CardManager{};
        sts::CardInstance defend{sts::CardId::DEFEND_RED}; defend.uniqueId = 1;
        state.cards.drawPile.push_back(defend); state.cards.nextUniqueCardId = 2;
        state.potionCount = state.potionCapacity = 0; state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
        TurnSearchCaps caps; caps.max_seconds = 10;
        auto evaluate = [](std::span<const Inputs> batch) {
            return std::vector<Prediction>(batch.size(), Prediction{20, {}});
        };
        TurnSearch one{state, caps}; const auto sequence = one.decide(1, evaluate);
        require(!one.stats().fallback && one.stats().expansions == 1 && one.root().children.size() == 1, "E=1 budget/exhaustive failure");
        for (const auto& child : one.root().children)
            require(turn_state_key(replay(state, child.sequence)) == child.key, "child key not reproduced");
        require(one.stats().max_depth == 1, "depth is not measured in turns");

        auto winning = state; winning.player.energy = 1;
        winning.cards.cardsInHand = 1; winning.cards.hand[0] = sts::CardInstance{sts::CardId::STRIKE_RED};
        winning.cards.hand[0].uniqueId = 2; winning.cards.nextUniqueCardId = 3; winning.cards.strikeCount = 1;
        winning.monsters.arr[0].curHp = 1;
        TurnSearch win{winning, caps}; const auto win_sequence = win.decide(1, evaluate);
        require(replay(winning, win_sequence).outcome == sts::Outcome::PLAYER_VICTORY, "E=1 did not choose exhaustive best terminal turn");
        double best = 0;
        for (const auto& child : win.root().children) {
            best = std::max(best, child.value);
            if (!child.terminal) require(turn_state_key(replay(winning, child.sequence)) == child.key, "winning fixture key mismatch");
        }
        require(best == 100, "terminal win not known 100");
        auto losing = winning; losing.player.curHp = 1;
        losing.cards.hand[0] = sts::CardInstance{sts::CardId::HEMOKINESIS};
        TurnSearch loss{losing, caps}; loss.decide(1, evaluate);
        bool known_loss = false;
        for (const auto& child : loss.root().children) if (child.terminal && child.key == "terminal:loss") known_loss |= child.value == 0;
        require(known_loss, "terminal loss not known zero");

        for (int which = 0; which < 5; ++which) {
            auto limited = caps;
            if (which == 0) limited.max_sequences = 1;
            if (which == 1) limited.root_max_children = 1;
            if (which == 2) limited.max_actions = 1;
            if (which == 3) limited.max_seconds = 1e-12;
            if (which == 4) limited.max_bytes = 1024;
            auto cap_state = winning;
            if (which == 2) cap_state.monsters.arr[0].curHp = 1000; // force a card + END_TURN path
            TurnSearch capped{cap_state, limited};
            require(capped.decide(8, evaluate).empty() && capped.stats().fallback && !capped.stats().reason.empty(), "root cap did not explicitly fall back");
            require(capped.root().children.empty(), "published partial expansion");
        }
        auto separated = caps; separated.root_max_children = 2; separated.max_children = 1;
        TurnSearch separate_root{winning, separated}; separate_root.decide(1, evaluate);
        require(!separate_root.stats().fallback && separate_root.root().children.size() == 2, "root did not use separate child cap");
        auto leaf_caps = caps; leaf_caps.max_children = 1;
        TurnSearch leaf{state, leaf_caps}; leaf.decide(8, evaluate);
        require(!leaf.stats().fallback && leaf.stats().expansions == 8, "nonroot cap/budget failure");
        require(leaf.root().children[0].node && leaf.root().children[0].node->capped &&
                leaf.root().children[0].node->children.empty(), "nonroot cap published children");
        require(leaf.root().children[0].q() == 20, "capped leaf lost own V");

        TurnSearch reuse{state, caps}; const auto chosen = reuse.decide(8, evaluate);
        require(reuse.advance(replay(state, chosen)), "exact subtree reuse failed");
        reuse.decide(1, evaluate); require(reuse.stats().reused, "reuse not reported");
        auto codex = state; codex.player.setHasRelic<sts::RelicId::NILRYS_CODEX>(true);
        TurnSearch choices{codex, caps}; choices.decide(1, evaluate);
        require(!choices.stats().fallback, "choice fixture capped");
        bool resolved = false;
        for (const auto& child : choices.root().children) {
            const auto next = replay(codex, child.sequence);
            require(child.terminal || (next.turn != codex.turn && next.inputState == sts::InputState::PLAYER_NORMAL), "keyed before end-turn choices resolved");
            if (!child.terminal) require(turn_state_key(next) == child.key, "choice child key mismatch");
            resolved |= child.sequence.size() > 1;
        }
        require(resolved, "END_TURN choice screen not enumerated");
        auto callbacks = state; callbacks.addToBot({[](sts::BattleContext&) {}});
        bool rejected = false; try { (void)turn_state_key(callbacks); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "callback endpoint key accepted");
        TurnSearch clamped{state, caps}; clamped.decide(1, [](std::span<const Inputs> batch) {
            return std::vector<Prediction>(batch.size(), Prediction{123, {}});
        }); require(clamped.root().children[0].value == 100, "child value not clamped");
        std::cout << "turn search tests passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
