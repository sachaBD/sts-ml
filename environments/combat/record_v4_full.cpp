#include "environments/combat/record_v4_full.hpp"

#include "combat/BattleContext.h"
#include "constants/MonsterEncounters.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Rooms.h"
#include "sim/search/Action.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <stdexcept>

namespace stsrl::combat_v4_full {

namespace {

namespace S = sts;
using sts::MonsterId;
using arrow::ArrayBuilder;
using arrow::field;

// ================= schema (kept equal to runs/schema=combat_v4_full/schema.py) =================

std::shared_ptr<arrow::Field> F(const char* name, std::shared_ptr<arrow::DataType> type, bool nullable = false) {
    return field(name, std::move(type), nullable);
}

using arrow::boolean; using arrow::int8; using arrow::int16; using arrow::int32; using arrow::uint8;
using arrow::uint32; using arrow::uint64; using arrow::float32; using arrow::utf8; using arrow::list;
using arrow::struct_;

const auto STATUS = struct_({F("status", utf8()), F("amount", int32())});
const auto CARD = struct_({F("card", utf8()), F("upgrades", int8()), F("cost", int8()), F("cost_for_turn", int8()),
                           F("special_data", int16()), F("free_to_play_once", boolean()), F("retain", boolean()),
                           F("unique_id", int16())});
const auto RELIC = struct_({F("relic", utf8()), F("counter", int16(), true)});
const auto MONSTER = struct_({F("slot", int8()), F("monster", utf8()), F("alive", boolean()), F("hp", int16()),
                              F("max_hp", int16()), F("block", int16()), F("intent", utf8(), true),
                              F("previous_move", utf8(), true), F("intent_damage", int16(), true),
                              F("intent_hits", int8(), true), F("statuses", list(STATUS)), F("counters", list(STATUS)),
                              F("half_dead", boolean()), F("escaping", boolean()), F("stasis", CARD, true)});
const auto PLAYER = struct_({F("hp", int16()), F("max_hp", int16()), F("block", int16()), F("energy", int8()),
                             F("energy_per_turn", int8()), F("draw_per_turn", int8()), F("gold", int16()),
                             F("statuses", list(STATUS)), F("cards_played_this_turn", int8()),
                             F("attacks_played_this_turn", int8()), F("skills_played_this_turn", int8()),
                             F("cards_discarded_this_turn", int8()), F("orange_pellets_types", uint8()),
                             F("times_damaged_this_combat", int16()), F("used_necronomicon_this_turn", boolean()),
                             F("combust_hp_loss", int8()), F("bombs", list(int16()))});
const auto PUBLIC = struct_({F("turn", int16()), F("player", PLAYER), F("relics", list(RELIC)),
                             F("potions", list(utf8())), F("hand", list(CARD)), F("draw", list(CARD)),
                             F("discard", list(CARD)), F("exhaust", list(CARD)), F("monsters", list(MONSTER))});
const auto RNG = struct_({F("stream", utf8()), F("counter", int32()), F("seed0", uint64()), F("seed1", uint64())});
const auto HIDDEN = struct_({F("draw_order", list(int16())), F("rng", list(RNG)),
                             F("hidden_intents", list(utf8()), true)});
const auto LEGAL = struct_({F("action", uint32()), F("visits", uint32(), true), F("value", float32(), true)});

// ================= builder helpers =================

void ok(const arrow::Status& s) {
    if (!s.ok()) throw std::runtime_error{"arrow: " + s.ToString()};
}
ArrayBuilder* child(ArrayBuilder* b, int i) { return static_cast<arrow::StructBuilder*>(b)->child_builder(i).get(); }
void begin(ArrayBuilder* b) { ok(static_cast<arrow::StructBuilder*>(b)->Append()); }
ArrayBuilder* items(ArrayBuilder* b) {  // start a list entry, return its element builder
    auto* l = static_cast<arrow::ListBuilder*>(b);
    ok(l->Append());
    return l->value_builder();
}
void null(ArrayBuilder* b) { ok(b->AppendNull()); }
void num(ArrayBuilder* b, std::int64_t v) {
    switch (b->type()->id()) {
        case arrow::Type::INT8: ok(static_cast<arrow::Int8Builder*>(b)->Append(static_cast<std::int8_t>(v))); break;
        case arrow::Type::INT16: ok(static_cast<arrow::Int16Builder*>(b)->Append(static_cast<std::int16_t>(v))); break;
        case arrow::Type::INT32: ok(static_cast<arrow::Int32Builder*>(b)->Append(static_cast<std::int32_t>(v))); break;
        case arrow::Type::UINT8: ok(static_cast<arrow::UInt8Builder*>(b)->Append(static_cast<std::uint8_t>(v))); break;
        case arrow::Type::UINT32: ok(static_cast<arrow::UInt32Builder*>(b)->Append(static_cast<std::uint32_t>(v))); break;
        case arrow::Type::UINT64: ok(static_cast<arrow::UInt64Builder*>(b)->Append(static_cast<std::uint64_t>(v))); break;
        default: throw std::logic_error{"num: unexpected builder type"};
    }
}
void real(ArrayBuilder* b, float v) { ok(static_cast<arrow::FloatBuilder*>(b)->Append(v)); }
void flag(ArrayBuilder* b, bool v) { ok(static_cast<arrow::BooleanBuilder*>(b)->Append(v)); }
void str(ArrayBuilder* b, const std::string& v) { ok(static_cast<arrow::StringBuilder*>(b)->Append(v)); }

std::unique_ptr<ArrayBuilder> make_root(const std::shared_ptr<arrow::Schema>& schema) {
    std::unique_ptr<ArrayBuilder> b;
    ok(arrow::MakeBuilder(arrow::default_memory_pool(), struct_(schema->fields()), &b));
    return b;
}

std::shared_ptr<arrow::Table> finish(ArrayBuilder& root, const std::shared_ptr<arrow::Schema>& schema) {
    std::shared_ptr<arrow::Array> array;
    ok(root.Finish(&array));
    auto batch = arrow::RecordBatch::FromStructArray(array);
    if (!batch.ok()) throw std::runtime_error{"arrow: " + batch.status().ToString()};
    auto table = arrow::Table::FromRecordBatches(schema, {*batch});
    if (!table.ok()) throw std::runtime_error{"arrow: " + table.status().ToString()};
    return *table;
}

// ================= names =================

std::string lower(const char* s) {
    std::string r{s};
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
}
template <typename Array>
std::vector<std::string> lower_table(const Array& names) {
    std::vector<std::string> r;
    for (const char* n : names) r.push_back(lower(n));
    return r;
}

struct Names {
    std::vector<std::string> card = lower_table(S::cardEnumStrings), relic = lower_table(S::relicEnumNames),
                             potion = lower_table(S::potionEnumNames), monster = lower_table(S::monsterIdStrings),
                             move = lower_table(S::monsterMoveStrings), mstatus = lower_table(S::monsterStatusEnumStrings),
                             pstatus = lower_table(::playerStatusEnumStrings),
                             encounter = lower_table(S::monsterEncounterEnumNames);
    std::vector<int> card_rank;  // position of each card in name order
    Names() {
        std::vector<int> ids(card.size());
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return card[a] < card[b]; });
        card_rank.resize(ids.size());
        for (std::size_t r = 0; r < ids.size(); ++r) card_rank[ids[r]] = static_cast<int>(r);
    }
};
const Names& names() {
    static const Names n;
    return n;
}

[[noreturn]] void undecoded(const std::string& message) { throw std::logic_error{"combat_v4_full: undecoded " + message}; }

// ================= cards, statuses =================

void put_status(ArrayBuilder* b, const std::string& name, int amount) {  // b: STATUS struct builder
    begin(b);
    str(child(b, 0), name);
    num(child(b, 1), amount);
}

void put_card(ArrayBuilder* b, const S::CardInstance& c) {
    begin(b);
    str(child(b, 0), names().card[static_cast<int>(c.id)]);
    num(child(b, 1), c.getUpgradeCount());
    num(child(b, 2), c.cost);
    num(child(b, 3), c.costForTurn);
    num(child(b, 4), c.specialData);
    flag(child(b, 5), c.freeToPlayOnce);
    flag(child(b, 6), c.retain);
    num(child(b, 7), c.uniqueId);
}

template <typename Iter>
void put_cards(ArrayBuilder* list_builder, Iter first, Iter last) {
    auto* cards = items(list_builder);
    for (; first != last; ++first) put_card(cards, *first);
}

// ================= player =================

void require_ironclad_only(const S::Player& p) {
    bool ok_ = p.stance == ::Stance::NEUTRAL && p.orbSlots == 0 && p.focus == 0 && p.lightningOrbsChanneled == 0 &&
               p.frostOrbsChanneled == 0;
    for (int i = 0; i < 10; ++i) ok_ = ok_ && p.orbs[i] == ::Orb::EMPTY && p.orbData[i] == 0;
    if (!ok_) throw std::logic_error{"combat_v4_full: non-default Defect / Watcher player field"};
}

void put_player_statuses(ArrayBuilder* list_builder, const S::Player& p) {
    auto* a = items(list_builder);
    for (int s = 1; s <= static_cast<int>(::PlayerStatus::THE_BOMB); ++s) {
        const auto ps = static_cast<::PlayerStatus>(s);
        if (ps == ::PlayerStatus::THE_BOMB || !p.hasStatusRuntime(ps)) continue;
        // flags live only in the status bits (no statusMap entry): amount 1
        const bool core = ps == ::PlayerStatus::ARTIFACT || ps == ::PlayerStatus::DEXTERITY ||
                          ps == ::PlayerStatus::STRENGTH || ps == ::PlayerStatus::FOCUS;
        const int amount = core || p.statusMap.count(ps) ? p.getStatusRuntime(ps) : 1;
        if (amount != 0) put_status(a, names().pstatus[s], amount);
    }
}

void put_player(ArrayBuilder* b, const S::BattleContext& bc) {
    const auto& p = bc.player;
    begin(b);
    num(child(b, 0), p.curHp); num(child(b, 1), p.maxHp); num(child(b, 2), p.block); num(child(b, 3), p.energy);
    num(child(b, 4), p.energyPerTurn); num(child(b, 5), p.cardDrawPerTurn); num(child(b, 6), p.gold);
    put_player_statuses(child(b, 7), p);
    num(child(b, 8), p.cardsPlayedThisTurn); num(child(b, 9), p.attacksPlayedThisTurn);
    num(child(b, 10), p.skillsPlayedThisTurn); num(child(b, 11), p.cardsDiscardedThisTurn);
    num(child(b, 12), static_cast<unsigned>(p.orangePelletsCardTypesPlayed.to_ulong()));
    num(child(b, 13), p.timesDamagedThisCombat); flag(child(b, 14), p.haveUsedNecronomiconThisTurn);
    num(child(b, 15), p.combustHpLoss);
    auto* bombs = items(child(b, 16));
    num(bombs, p.bomb1); num(bombs, p.bomb2); num(bombs, p.bomb3);
}

void put_relics(ArrayBuilder* list_builder, const S::GameContext& gc, const S::Player& p) {
    auto* a = items(list_builder);
    for (const auto& r : gc.relics.relics) {
        begin(a);
        str(child(a, 0), names().relic[static_cast<int>(r.id)]);
        bool has_counter = true;
        int counter = 0;
        switch (r.id) {
            case S::RelicId::PEN_NIB: counter = p.penNibCounter; break;
            case S::RelicId::NUNCHAKU: counter = p.nunchakuCounter; break;
            case S::RelicId::INK_BOTTLE: counter = p.inkBottleCounter; break;
            case S::RelicId::HAPPY_FLOWER: counter = p.happyFlowerCounter; break;
            case S::RelicId::INCENSE_BURNER: counter = p.incenseBurnerCounter; break;
            case S::RelicId::SUNDIAL: counter = p.sundialCounter; break;
            default: has_counter = false; break;
        }
        if (has_counter) num(child(a, 1), counter);
        else null(child(a, 1));
    }
}

// ================= monsters =================

struct Raw {
    const char* misc = nullptr;  // what Monster::miscInfo holds for this monster, or nullptr
    const char* up0 = nullptr;   // Monster::uniquePower0 (outside the unique-power statuses)
    const char* up1 = nullptr;   // Monster::uniquePower1
};

// The per-MonsterId decoding of the raw integers (sts_lightspeed src/combat/MonsterSpecific.cpp, Monster::construct).
// Monsters not listed have no raw state outside the statuses (a non-zero raw value there is undecoded: crash).
Raw raw_names(MonsterId id) {
    switch (id) {
        case MonsterId::BOOK_OF_STABBING: return {"stab_count"};
        case MonsterId::THE_GUARDIAN: return {"mode_shift_threshold"};
        case MonsterId::BRONZE_AUTOMATON: return {"last_boost_was_flail"};
        case MonsterId::BRONZE_ORB: return {"used_stasis"};
        case MonsterId::GREMLIN_WIZARD: return {"charge"};
        case MonsterId::HEXAGHOST: return {"divider_damage", "orb_count"};
        case MonsterId::SPIKER: return {"thorns_used"};
        case MonsterId::DARKLING: return {"nip_damage"};
        case MonsterId::WRITHING_MASS: return {"used_implant"};
        case MonsterId::TIME_EATER: return {"used_haste"};
        case MonsterId::AWAKENED_ONE: return {"phase2"};
        case MonsterId::RED_SLAVER: return {"used_entangle"};
        case MonsterId::GREEN_LOUSE:
        case MonsterId::RED_LOUSE: return {"bite_damage", "curl_up_raw"};  // curl up spent: amount stays, flag clears
        case MonsterId::LOOTER:
        case MonsterId::MUGGER: return {"stolen_gold"};
        case MonsterId::THE_CHAMP: return {"champ_misc"};  // split below
        default: return {};
    }
}

bool is_up0_status(S::MonsterStatus s) { return s >= S::MS::ANGRY && s <= S::MS::TIME_WARP; }
bool is_up1_status(S::MonsterStatus s) { return s >= S::MS::INVINCIBLE && s <= S::MS::SHARP_HIDE; }
bool is_flag_status(S::MonsterStatus s) { return s >= S::MS::ASLEEP && s <= S::MS::STASIS; }

void put_monster(ArrayBuilder* b, const S::BattleContext& bc, const S::Monster& m, int slot, bool dome) {
    const std::string& name = names().monster[static_cast<int>(m.id)];
    begin(b);
    num(child(b, 0), slot);
    str(child(b, 1), name);
    flag(child(b, 2), m.isTargetable());
    num(child(b, 3), m.curHp); num(child(b, 4), m.maxHp); num(child(b, 5), m.block);

    // intent
    if (m.moveHistory[0] != S::MMID::INVALID && !dome) {
        str(child(b, 6), names().move[static_cast<int>(m.moveHistory[0])]);
    } else {
        null(child(b, 6));
    }
    if (m.moveHistory[1] != S::MMID::INVALID) {
        str(child(b, 7), names().move[static_cast<int>(m.moveHistory[1])]);
    } else {
        null(child(b, 7));
    }
    if (m.moveHistory[0] != S::MMID::INVALID && !dome && m.isAttacking()) {
        const auto base = m.getMoveBaseDamage(bc);
        num(child(b, 8), m.calculateDamageToPlayer(bc, base.damage));
        num(child(b, 9), base.attackCount);
    } else {
        null(child(b, 8));
        null(child(b, 9));
    }

    // statuses
    auto* statuses = items(child(b, 10));
    bool up0_explained = false, up1_explained = false;
    for (int s = 0; s < static_cast<int>(S::MS::INVALID); ++s) {
        const auto ms = static_cast<S::MonsterStatus>(s);
        const bool has = m.hasStatusInternal(ms);
        const int amount = is_flag_status(ms) ? (has ? 1 : 0) : m.getStatusInternal(ms);
        if (has && is_up0_status(ms)) up0_explained = true;
        if (has && is_up1_status(ms)) up1_explained = true;
        if (amount != 0) put_status(statuses, names().mstatus[s], amount);
    }

    // counters
    auto* counters = items(child(b, 11));
    const Raw raw = raw_names(m.id);
    if (m.id == MonsterId::THE_CHAMP) {  // miscInfo: bits 0-1 defensive stance uses, bit 2 phase 2 (enraged)
        put_status(counters, "defensive_stances_used", m.miscInfo & 0x3);
        put_status(counters, "phase2", (m.miscInfo >> 2) & 1);
        if (m.miscInfo & ~0x7) undecoded("the_champ miscInfo bits");
    } else if (raw.misc) {
        put_status(counters, raw.misc, m.miscInfo);
    } else if (m.miscInfo != 0) {
        undecoded("miscInfo on " + name);
    }
    if (raw.up0) {
        put_status(counters, raw.up0, m.uniquePower0);
    } else if (m.uniquePower0 != 0 && !up0_explained) {
        undecoded("uniquePower0 on " + name);
    }
    if (raw.up1) {
        put_status(counters, raw.up1, m.uniquePower1);
    } else if (m.uniquePower1 != 0 && !up1_explained) {
        undecoded("uniquePower1 on " + name);
    }

    flag(child(b, 12), m.halfDead);
    flag(child(b, 13), m.isEscapingB || m.escapeNext);
    if (m.id == MonsterId::BRONZE_ORB && m.miscInfo && bc.cards.stasisCards[std::min(1, m.idx)].id != S::CardId::INVALID) {
        put_card(child(b, 14), bc.cards.stasisCards[std::min(1, m.idx)]);
    } else {
        null(child(b, 14));
    }
}

// ================= state =================

int potion_count(const S::BattleContext& bc) {
    int n = 0;
    for (int i = 0; i < bc.potionCapacity; ++i)
        n += bc.potions[i] != S::Potion::EMPTY_POTION_SLOT && bc.potions[i] != S::Potion::INVALID;
    return n;
}

const char* kind_of(const S::BattleContext& bc) {
    if (bc.inputState == S::InputState::PLAYER_NORMAL) return "play";
    if (bc.inputState == S::InputState::CARD_SELECT) {
        static const auto tasks = lower_table(S::cardSelectTaskStrings);
        return tasks[static_cast<int>(bc.cardSelectInfo.cardSelectTask)].c_str();
    }
    throw std::logic_error{"combat_v4_full: undecoded input state " + std::to_string(static_cast<int>(bc.inputState))};
}

std::string room_name(int room) {
    switch (static_cast<S::Room>(room)) {
        case S::Room::MONSTER: return "monster";
        case S::Room::ELITE: return "elite";
        case S::Room::BOSS: return "boss";
        case S::Room::EVENT: return "event";
        default: return lower(S::roomStrings[room]);
    }
}

void put_rng(ArrayBuilder* b, const char* name, const S::Random& r) {
    begin(b);
    str(child(b, 0), name);
    num(child(b, 1), r.counter); num(child(b, 2), static_cast<std::int64_t>(r.seed0));
    num(child(b, 3), static_cast<std::int64_t>(r.seed1));
}

}  // namespace

std::shared_ptr<arrow::Schema> decisions_schema() {
    return arrow::schema({F("fight_id", utf8()), F("step", int16()), F("kind", utf8()), F("public", PUBLIC),
                          F("hidden", HIDDEN), F("legal", list(LEGAL)), F("chosen", int16()), F("explored", boolean()),
                          F("root_value", float32(), true), F("simulations", uint32(), true)});
}

std::shared_ptr<arrow::Schema> fights_schema() {
    return arrow::schema({F("fight_id", utf8()), F("encounter", utf8()), F("act", int8()), F("floor", int16()),
                          F("ascension", int8()), F("room", utf8()), F("agent", utf8()), F("decisions", int16()),
                          F("turns", int16()), F("won", boolean()), F("start_hp", int16()), F("final_hp", int16()),
                          F("max_hp", int16()), F("potions_start", int8()), F("potions_end", int8()),
                          F("monster_hp_left", int16())});
}

struct Expander::Impl {
    std::shared_ptr<arrow::Schema> dschema = decisions_schema(), fschema = fights_schema();
    std::unique_ptr<ArrayBuilder> d = make_root(dschema), f = make_root(fschema);
    std::int64_t n_decisions = 0, n_fights = 0;

    // Replay without emitting: the checks that make add() return false.
    bool validate(const Fight& fight, const S::GameContext& gc, std::string& why) const {
        S::BattleContext bc;
        bc.init(gc);
        for (std::size_t i = 0; i < fight.actions.size(); ++i) {
            if (bc.outcome != S::Outcome::UNDECIDED) { why = "fight ended before action " + std::to_string(i); return false; }
            const S::search::Action action{fight.actions[i]};
            if (!action.isValidAction(bc)) { why = "invalid action " + std::to_string(i); return false; }
            const auto legal = S::search::Action::getAllActionsInState(bc);
            if (legal.empty()) throw std::logic_error{"combat_v4_full: no legal action at step " + std::to_string(i)};
            if (std::none_of(legal.begin(), legal.end(), [&](const auto& a) { return a.bits == fight.actions[i]; })) {
                why = "chosen action not among the legal moves at step " + std::to_string(i);
                return false;
            }
            action.execute(bc);
        }
        if (bc.outcome == S::Outcome::UNDECIDED) { why = "fight did not end"; return false; }
        if ((bc.outcome == S::Outcome::PLAYER_VICTORY) != fight.won || bc.player.curHp != fight.final_hp) {
            why = "won / final_hp differ from the recording";
            return false;
        }
        return true;
    }

    void put_decision(const Fight& fight, const S::BattleContext& bc, const S::GameContext& gc,
                      const std::vector<S::search::Action>& legal, std::size_t step) {
        require_ironclad_only(bc.player);
        const bool dome = bc.player.hasRelic<S::RelicId::RUNIC_DOME>();
        begin(d.get());
        str(child(d.get(), 0), fight.fight_id);
        num(child(d.get(), 1), static_cast<std::int64_t>(step));
        str(child(d.get(), 2), kind_of(bc));

        // public
        auto* pub = child(d.get(), 3);
        begin(pub);
        num(child(pub, 0), bc.turn);
        put_player(child(pub, 1), bc);
        put_relics(child(pub, 2), gc, bc.player);
        auto* potions = items(child(pub, 3));
        for (int i = 0; i < bc.potionCapacity; ++i) {
            const auto p = bc.potions[i];
            if (p == S::Potion::EMPTY_POTION_SLOT || p == S::Potion::INVALID) null(potions);
            else str(potions, names().potion[static_cast<int>(p)]);
        }
        put_cards(child(pub, 4), bc.cards.hand.begin(), bc.cards.hand.begin() + bc.cards.cardsInHand);
        {  // draw pile as a multiset: sorted by card name, upgrades, unique id
            std::vector<const S::CardInstance*> v;
            for (const auto& c : bc.cards.drawPile) v.push_back(&c);
            const auto& rank = names().card_rank;
            std::sort(v.begin(), v.end(), [&](const auto* a, const auto* b) {
                const int ra = rank[static_cast<int>(a->id)], rb = rank[static_cast<int>(b->id)];
                if (ra != rb) return ra < rb;
                if (a->getUpgradeCount() != b->getUpgradeCount()) return a->getUpgradeCount() < b->getUpgradeCount();
                return a->uniqueId < b->uniqueId;
            });
            auto* cards = items(child(pub, 5));
            for (const auto* c : v) put_card(cards, *c);
        }
        put_cards(child(pub, 6), bc.cards.discardPile.begin(), bc.cards.discardPile.end());
        put_cards(child(pub, 7), bc.cards.exhaustPile.begin(), bc.cards.exhaustPile.end());
        auto* monsters = items(child(pub, 8));
        for (int i = 0; i < 5; ++i) {
            const auto& m = bc.monsters.arr[i];
            if (m.id != MonsterId::INVALID) put_monster(monsters, bc, m, i, dome);
        }

        // hidden
        auto* hid = child(d.get(), 4);
        begin(hid);
        auto* order = items(child(hid, 0));
        for (auto it = bc.cards.drawPile.rbegin(); it != bc.cards.drawPile.rend(); ++it) num(order, it->uniqueId);
        auto* rng = items(child(hid, 1));
        put_rng(rng, "ai", bc.aiRng); put_rng(rng, "card_random", bc.cardRandomRng); put_rng(rng, "misc", bc.miscRng);
        put_rng(rng, "monster_hp", bc.monsterHpRng); put_rng(rng, "potion", bc.potionRng);
        put_rng(rng, "shuffle", bc.shuffleRng);
        if (dome) {
            auto* hidden = items(child(hid, 2));
            for (int i = 0; i < 5; ++i) {
                const auto& m = bc.monsters.arr[i];
                if (m.id == MonsterId::INVALID) continue;
                if (m.moveHistory[0] == S::MMID::INVALID) null(hidden);
                else str(hidden, names().move[static_cast<int>(m.moveHistory[0])]);
            }
        } else {
            null(child(hid, 2));
        }

        // legal, chosen, search
        const SearchStep* search = fight.search.empty() ? nullptr : fight.search[step];
        auto* moves = items(child(d.get(), 5));
        int chosen = -1;
        for (std::size_t i = 0; i < legal.size(); ++i) {
            begin(moves);
            num(child(moves, 0), legal[i].bits);
            const SearchChild* c = nullptr;
            if (search)
                for (const auto& k : search->children)
                    if (k.action == legal[i].bits) { c = &k; break; }
            if (c) { num(child(moves, 1), c->visits); real(child(moves, 2), c->value); }
            else { null(child(moves, 1)); null(child(moves, 2)); }
            if (chosen < 0 && legal[i].bits == fight.actions[step]) chosen = static_cast<int>(i);
        }
        num(child(d.get(), 6), chosen);
        flag(child(d.get(), 7), fight.explored[step] != 0);
        if (search) { real(child(d.get(), 8), search->root_value); num(child(d.get(), 9), search->simulations); }
        else { null(child(d.get(), 8)); null(child(d.get(), 9)); }
        ++n_decisions;
    }
};

Expander::Expander() : impl_(std::make_unique<Impl>()) {}
Expander::~Expander() = default;
std::int64_t Expander::decisions() const { return impl_->n_decisions; }
std::int64_t Expander::fights() const { return impl_->n_fights; }
std::shared_ptr<arrow::Table> Expander::take_decisions() {
    auto t = finish(*impl_->d, impl_->dschema);
    impl_->d = make_root(impl_->dschema);
    return t;
}
std::shared_ptr<arrow::Table> Expander::take_fights() {
    auto t = finish(*impl_->f, impl_->fschema);
    impl_->f = make_root(impl_->fschema);
    return t;
}

bool Expander::add(const Fight& fight, std::string& mismatch) {
    auto& I = *impl_;
    const auto gc = combat_v4::start_game(fight.start);
    if (!I.validate(fight, gc, mismatch)) return false;

    S::BattleContext bc;
    bc.init(gc);
    const int start_hp = bc.player.curHp, potions_start = potion_count(bc);
    for (std::size_t i = 0; i < fight.actions.size(); ++i) {
        const auto legal = S::search::Action::getAllActionsInState(bc);
        I.put_decision(fight, bc, gc, legal, i);
        S::search::Action{fight.actions[i]}.execute(bc);
    }
    int hp_left = 0;
    for (int i = 0; i < 5; ++i) {
        const auto& m = bc.monsters.arr[i];
        if (m.id != MonsterId::INVALID && m.curHp > 0) hp_left += m.curHp;
    }
    auto* f = I.f.get();
    const auto& st = fight.start;
    begin(f);
    str(child(f, 0), fight.fight_id);
    str(child(f, 1), names().encounter[st.encounter]);
    num(child(f, 2), st.act); num(child(f, 3), st.floor); num(child(f, 4), st.ascension);
    str(child(f, 5), room_name(st.cur_room));
    str(child(f, 6), fight.agent);
    num(child(f, 7), static_cast<std::int64_t>(fight.actions.size()));
    num(child(f, 8), bc.turn);
    flag(child(f, 9), bc.outcome == S::Outcome::PLAYER_VICTORY);
    num(child(f, 10), start_hp); num(child(f, 11), bc.player.curHp); num(child(f, 12), bc.player.maxHp);
    num(child(f, 13), potions_start); num(child(f, 14), potion_count(bc)); num(child(f, 15), hp_left);
    ++I.n_fights;
    return true;
}

}  // namespace stsrl::combat_v4_full
