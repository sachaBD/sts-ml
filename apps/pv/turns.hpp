// Standalone turn-enumeration measurement, not used by PV search.
#pragma once
#include "agents/combat/pv/search.hpp"
#include "agents/combat/pv/turn_state_key.hpp"
#include "environments/combat/record_v4.hpp"
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <type_traits>
#include <openssl/sha.h>
#include <cstring>
#include <nlohmann/json.hpp>

namespace stsrl::pv {

inline void turn_key_sanity(sts::BattleContext state) {
    state.inputState = sts::InputState::PLAYER_NORMAL;
    state.actionQueue.clear(); state.cardQueue.clear();
    // Non-interacting card fixture, derived from the same real RNG/monster context.
    state.player = sts::Player{};
    state.player.cc = sts::CharacterClass::IRONCLAD;
    state.player.energy = 3;
    state.cards = sts::CardManager{};
    state.cards.cardsInHand = 2;
    state.cards.hand[0] = sts::CardInstance{sts::CardId::DEFEND_RED}; state.cards.hand[0].uniqueId = 0;
    state.cards.hand[1] = sts::CardInstance{sts::CardId::STRIKE_RED}; state.cards.hand[1].uniqueId = 1;
    for (int i = 2; i < 32; ++i) {
        sts::CardInstance card{sts::CardId::DEFEND_RED}; card.uniqueId = i;
        state.cards.drawPile.push_back(card);
    }
    state.cards.nextUniqueCardId = 32;
    state.cards.strikeCount = 1;
    state.monsters.arr[0].curHp = state.monsters.arr[0].maxHp = 1000; // no phase transition or kill
    state.monsterTurnIdx = 6; state.endTurnQueued = false; state.turnHasEnded = false;
    state.endTurnAfterCurrentCard = false;
    // Two non-interacting potion effects commute, unlike card plays that reorder the discard pile.
    state.potionCapacity = std::max(2, state.potionCapacity);
    state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
    state.potions[0] = sts::Potion::BLOCK_POTION; state.potions[1] = sts::Potion::STRENGTH_POTION;
    state.potionCount = 2;
    const sts::search::Action a{sts::search::ActionType::POTION, 0, 0};
    const sts::search::Action b{sts::search::ActionType::POTION, 1, 0};
    auto execute = [&](bool reverse) {
        auto copy = state;
        for (const auto action : reverse ? std::array{b, a} : std::array{a, b}) {
            if (!action.isValidAction(copy)) throw std::runtime_error{"turns: sanity potion is illegal input=" + std::to_string(int(copy.inputState)) + " slot=" + std::to_string(action.getSourceIdx())};
            action.execute(copy);
        }
        return copy;
    };
    const auto first = execute(false), repeat = execute(false), reversed = execute(true);
    const auto key = turn_state_key(first);
    if (key != turn_state_key(repeat)) throw std::runtime_error{"turns: identical-sequence key sanity failed"};
    if (key != turn_state_key(reversed)) {
        std::vector<std::pair<std::size_t, std::string>> fields;
        (void)turn_state_key(first, false, &fields);
        const auto other = turn_state_key(reversed);
        const auto at = std::mismatch(key.begin(), key.end(), other.begin(), other.end()).first - key.begin();
        std::string field;
        for (const auto& [offset, name] : fields) if (offset <= std::size_t(at)) field = name;
        throw std::runtime_error{"turns: commuting-potion key sanity failed field=" + field};
    }

    auto play_pair = [&](bool reverse) {
        auto copy = state;
        for (auto id : reverse ? std::array{sts::CardId::STRIKE_RED, sts::CardId::DEFEND_RED} :
                                 std::array{sts::CardId::DEFEND_RED, sts::CardId::STRIKE_RED}) {
            int slot = 0;
            while (slot < copy.cards.cardsInHand && copy.cards.hand[slot].id != id) ++slot;
            const sts::search::Action action{sts::search::ActionType::CARD, slot, 0};
            if (!action.isValidAction(copy)) throw std::runtime_error{"turns: sanity card is illegal"};
            action.execute(copy);
        }
        const sts::search::Action end{sts::search::ActionType::END_TURN};
        if (!end.isValidAction(copy)) throw std::runtime_error{"turns: sanity END_TURN is illegal"};
        end.execute(copy);
        return copy;
    };
    const auto ds = play_pair(false), sd = play_pair(true);
    if (turn_state_key(ds, true) != turn_state_key(sd, true))
        throw std::runtime_error{"turns: canonical commuting-card key sanity failed"};
    if (turn_state_key(ds) == turn_state_key(sd))
        throw std::runtime_error{"turns: exact key incorrectly ignored discard order"};
}

using TurnDigest = std::array<std::uint64_t, 2>;
inline TurnDigest turn_digest(const std::string& key) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(key.data()), key.size(), bytes);
    TurnDigest digest;
    std::memcpy(digest.data(), bytes, sizeof(digest)); // first 128 SHA-256 bits
    return digest;
}
struct TurnDigestHash {
    std::size_t operator()(const TurnDigest& digest) const { return digest[0] ^ digest[1]; }
};

struct TurnDifferential {
    int tested = 0, divergent = 0, illegal = 0, capped = 0;

    void check(sts::BattleContext a, sts::BattleContext b) {
        ++tested;
        for (int step = 0; step < 512; ++step) {
            const auto moves = sts::search::Action::getAllActionsInState(a);
            if (moves.empty()) { ++illegal; return; }
            const auto action = moves.front(); // deterministic first-legal next-turn continuation
            auto mapped = action;
            if (action.getActionType() == sts::search::ActionType::CARD) {
                const auto id = a.cards.hand[action.getSourceIdx()].uniqueId;
                int slot = 0;
                while (slot < b.cards.cardsInHand && b.cards.hand[slot].uniqueId != id) ++slot;
                if (slot == b.cards.cardsInHand) { ++illegal; return; }
                mapped = sts::search::Action{sts::search::ActionType::CARD, slot, action.getTargetIdx()};
            } // potion slots/targets and selection bits stay unchanged; illegal replay is reported separately.
            if (!mapped.isValidAction(b)) { ++illegal; return; }
            const int turn = a.turn;
            action.execute(a); mapped.execute(b);
            if (a.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE ||
                b.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE)
                throw std::runtime_error{"turns: unsupported differential effect"};
            const bool done_a = a.outcome != sts::Outcome::UNDECIDED;
            const bool done_b = b.outcome != sts::Outcome::UNDECIDED;
            if (done_a != done_b || a.turn != b.turn) { ++divergent; return; }
            if (done_a) { divergent += a.outcome != b.outcome; return; } // known-value terminal leaves need no key
            if (action.getActionType() == sts::search::ActionType::END_TURN || done_a || a.turn != turn) {
                divergent += turn_state_key(a, true) != turn_state_key(b, true);
                return;
            }
        }
        ++capped;
    }
};

struct TurnEnumeration {
    static constexpr std::uint64_t cap = 1'000'000;
    std::uint64_t sequences = 0, terminal_wins = 0, terminal_losses = 0;
    bool capped = false;
    std::string cap_reason;
    std::unordered_set<TurnDigest, TurnDigestHash> states, canonical_states;
    struct Representative { TurnDigest exact; sts::BattleContext state; };
    std::unordered_map<TurnDigest, Representative, TurnDigestHash> representatives;
    TurnDifferential& differential;
    int pairs = 0;
    explicit TurnEnumeration(TurnDifferential& d) : differential{d} {}

    void endpoint(const sts::BattleContext& next) {
        if (next.outcome != sts::Outcome::UNDECIDED) {
            next.outcome == sts::Outcome::PLAYER_VICTORY ? ++terminal_wins : ++terminal_losses;
            return; // terminal endpoints excluded from distinct-state counts
        }
        const auto exact = turn_digest(turn_state_key(next)), canonical = turn_digest(turn_state_key(next, true));
        states.insert(exact); canonical_states.insert(canonical);
        // Conservative storage estimate includes hash nodes/buckets, not just the 16-byte payload.
        const auto bytes = (states.size() + canonical_states.size()) * 64 +
                           representatives.size() * (sizeof(sts::BattleContext) + 128);
        if (bytes >= (1ULL << 30)) { capped = true; cap_reason = "memory"; return; }
        if (differential.tested >= 200 || pairs >= 5) return;
        const auto found = representatives.find(canonical);
        if (found != representatives.end()) {
            if (exact != found->second.exact) { differential.check(found->second.state, next); ++pairs; }
        } else if (representatives.size() < 64) {
            representatives.emplace(canonical, Representative{exact, next});
        }
    }

    void dfs(const sts::BattleContext& state, int depth = 0) {
        if (capped) return;
        if (depth >= 512) { capped = true; cap_reason = "depth"; return; } // zero-cost cycles cannot recurse forever
        const auto moves = sts::search::Action::getAllActionsInState(state);
        if (moves.empty()) throw std::runtime_error{"turns: no legal moves in an undecided state"};
        for (const auto action : moves) {
            if (capped) return;
            if (!action.isValidAction(state)) throw std::runtime_error{"turns: enumerated illegal action"};
            auto next = state;
            action.execute(next);
            if (next.unsupportedEffectKind != sts::UnsupportedEffectKind::NONE)
                throw std::runtime_error{"turns: unsupported simulator effect"};
            if (action.getActionType() == sts::search::ActionType::END_TURN ||
                next.outcome != sts::Outcome::UNDECIDED || next.turn != state.turn) {
                endpoint(next);
                if (++sequences == cap) { capped = true; cap_reason = "sequences"; return; }
            } else {
                dfs(next, depth + 1);
            }
        }
    }
};

inline void measure_turns(const nlohmann::json& fight) {
    const auto actions = fight.at("actions").get<std::vector<std::uint32_t>>();
    const auto rebuilt = stsrl::combat_v4::replay(fight.at("start"), actions);
    if ((rebuilt.outcome == sts::Outcome::PLAYER_VICTORY) != fight.at("won").get<bool>() ||
        rebuilt.player.curHp != fight.at("final_hp").get<int>())
        throw std::runtime_error{"turns: recorded outcome mismatch"};
    sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
    turn_key_sanity(state);
    static TurnDifferential differential;
    int previous_turn = -1;
    for (auto bits : actions) {
        if (state.turn != previous_turn) {
            previous_turn = state.turn;
            const auto started = std::chrono::steady_clock::now();
            const auto before = differential;
            TurnEnumeration enumeration{differential};
            enumeration.dfs(state);
            const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << nlohmann::json{{"fight_id", fight.at("fight_id")}, {"turn", state.turn},
                {"sequences", enumeration.sequences}, {"distinct", enumeration.states.size()}, {"canonical_distinct", enumeration.canonical_states.size()},
                {"capped", enumeration.capped}, {"cap_reason", enumeration.cap_reason}, {"seconds", seconds},
                {"terminal_wins", enumeration.terminal_wins}, {"terminal_losses", enumeration.terminal_losses},
                {"pairs", differential.tested - before.tested}, {"divergent", differential.divergent - before.divergent},
                {"illegal", differential.illegal - before.illegal}, {"pair_capped", differential.capped - before.capped}}.dump() << std::endl;
        }
        const sts::search::Action action{bits};
        if (!action.isValidAction(state)) throw std::runtime_error{"turns: recorded prefix is illegal"};
        action.execute(state);
    }
    std::cout << nlohmann::json{{"done", fight.at("fight_id")}, {"sanity", "passed"}}.dump() << std::endl;
}

} // namespace stsrl::pv
