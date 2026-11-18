#include "environments/combat/state_codec.hpp"
#include <bitset>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace stsrl::combat {
namespace {
using namespace sts;

// Every field is listed by hand below; a simulator change that resizes these structs must be reflected here.
static_assert(sizeof(BattleContext) == 3488 && sizeof(Player) == 216 && sizeof(Monster) == 80 &&
              sizeof(CardInstance) == 12 && sizeof(CardQueueItem) == 36 && sizeof(CardSelectInfo) == 20);

template <class T> std::int64_t value(const T& v) { return static_cast<std::int64_t>(v); }

// A field and its default: groups write only the fields that differ from theirs.
template <class T> struct Field { T& v; std::int64_t d; };
template <class T> Field<T> f(T& v, std::int64_t d = 0) { return {v, d}; }

// Writer and Reader share one routine (code), so the two directions cannot drift apart.
struct Writer {
    static constexpr bool reading = false;
    std::string& out;
    void uv(std::uint64_t v) {
        for (; v >= 0x80; v >>= 7) out.push_back(char(v | 0x80));
        out.push_back(char(v));
    }
    void sv(std::int64_t v) { uv((std::uint64_t(v) << 1) ^ std::uint64_t(v >> 63)); }  // zigzag
    void varint(std::uint64_t& v) { uv(v); }  // v, read or written
    template <class T> void u(T& v) { uv(std::uint64_t(value(v))); }
    template <class T> void s(T& v) { sv(value(v)); }
    template <class T> void raw(T& v) { out.append(reinterpret_cast<const char*>(&v), sizeof v); }
    template <class... T> void group(Field<T>... fields) {
        std::uint64_t mask = 0;
        int i = 0;
        ((mask |= std::uint64_t(value(fields.v) != fields.d) << i++), ...);
        uv(mask);
        ((value(fields.v) != fields.d ? sv(value(fields.v) - fields.d) : void()), ...);
    }
};

struct Reader {
    static constexpr bool reading = true;
    const char* p;
    const char* end;
    std::uint64_t uv() {
        std::uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p == end) throw std::invalid_argument{"unpack: truncated state"};
            const auto byte = static_cast<unsigned char>(*p++);
            v |= std::uint64_t(byte & 0x7F) << shift;
            if (!(byte & 0x80)) return v;
        }
        throw std::invalid_argument{"unpack: bad varint"};
    }
    std::int64_t sv() {
        const auto z = uv();
        return std::int64_t(z >> 1) ^ -std::int64_t(z & 1);
    }
    void varint(std::uint64_t& v) { v = uv(); }
    template <class T> void u(T& v) { v = static_cast<T>(uv()); }
    template <class T> void s(T& v) { v = static_cast<T>(sv()); }
    template <class T> void raw(T& v) {
        if (end - p < std::ptrdiff_t(sizeof v)) throw std::invalid_argument{"unpack: truncated state"};
        std::memcpy(&v, p, sizeof v);
        p += sizeof v;
    }
    template <class... T> void group(Field<T>... fields) {
        const auto mask = uv();
        int i = 0;
        ((fields.v = static_cast<T>((mask >> i++ & 1) ? fields.d + sv() : fields.d)), ...);
    }
};

// A card is its id, upgrade and uniqueId; anything else only where it differs from a fresh card of that id.
template <class C> void card(C& c, CardInstance& x) {
    std::uint64_t word = 0;
    if constexpr (!C::reading) {
        const CardInstance fresh{x.id, x.upgraded};
        const bool extra = x.specialData != fresh.specialData || x.cost != fresh.cost ||
                           x.costForTurn != fresh.costForTurn || x.freeToPlayOnce || x.retain;
        word = std::uint64_t(x.id) << 2 | std::uint64_t(x.upgraded) << 1 | extra;
    }
    c.varint(word);
    if constexpr (C::reading) x = CardInstance{CardId(word >> 2), bool(word >> 1 & 1)};
    c.s(x.uniqueId);
    if (word & 1) {
        const CardInstance fresh{x.id, x.upgraded};
        c.group(f(x.specialData, fresh.specialData), f(x.cost, fresh.cost), f(x.costForTurn, fresh.costForTurn),
                f(x.freeToPlayOnce), f(x.retain));
    }
}

template <class C> void cards(C& c, std::vector<CardInstance>& pile) {
    std::uint64_t n = pile.size();
    c.varint(n);
    if constexpr (C::reading) pile.resize(n);
    for (auto& x : pile) card(c, x);
}

// Slots of a fixed array that hold a card, as a bit mask, then those cards.
template <class C, std::size_t N> void card_slots(C& c, std::array<CardInstance, N>& slots) {
    std::uint64_t used = 0;
    for (std::size_t i = 0; i < N; ++i) used |= std::uint64_t(slots[i].id != CardId::INVALID) << i;
    c.varint(used);
    for (std::size_t i = 0; i < N; ++i)
        if (used >> i & 1) card(c, slots[i]);
}

template <class C> void queue_item(C& c, CardQueueItem& q) {
    static const CardQueueItem d{};
    card(c, q.card);
    c.group(f(q.target, d.target), f(q.isEndTurn, d.isEndTurn), f(q.triggerOnUse, d.triggerOnUse),
            f(q.ignoreEnergyTotal, d.ignoreEnergyTotal), f(q.energyOnUse, d.energyOnUse), f(q.freeToPlay, d.freeToPlay),
            f(q.randomTarget, d.randomTarget), f(q.autoplay, d.autoplay), f(q.regretCardCount, d.regretCardCount),
            f(q.purgeOnUse, d.purgeOnUse), f(q.exhaustOnUse, d.exhaustOnUse));
}

// The combat RNGs all begin at Random(seed + floor) (BattleContext::init), so each is its number of steps from
// there: about two bytes instead of sixteen. One not reached within the search limit is written in full.
constexpr std::uint64_t rng_search_limit = 1 << 16;
template <class C> void combat_rng(C& c, Random& r, const Random& start) {
    std::uint64_t tag = 1;  // low bit set: written in full
    if constexpr (!C::reading) {
        Random walk = start;
        for (std::uint64_t n = 0; n <= rng_search_limit; ++n, walk.nextLong())
            if (walk.seed0 == r.seed0 && walk.seed1 == r.seed1) { tag = n << 1; break; }
    }
    c.varint(tag);
    if (tag & 1) {
        c.raw(r.seed0);
        c.raw(r.seed1);
    } else if constexpr (C::reading) {
        r = start;
        for (std::uint64_t n = 0; n < tag >> 1; ++n) r.nextLong();
    }
    c.s(r.counter);
}

template <class C> void code(C& c, BattleContext& bc) {
    static const BattleContext d{};
    c.u(bc.seed);
    c.s(bc.floorNum);
    c.u(bc.encounter);
    c.u(bc.ascension);
    const Random start{bc.seed + bc.floorNum};
    for (Random* r : {&bc.aiRng, &bc.cardRandomRng, &bc.monsterHpRng, &bc.shuffleRng}) combat_rng(c, *r, start);
    for (Random* r : {&bc.miscRng, &bc.potionRng}) {  // carried over from the run, so written in full
        c.s(r->counter);
        c.raw(r->seed0);
        c.raw(r->seed1);
    }

    std::uint64_t misc_bits = bc.miscBits.to_ullong();
    c.group(f(bc.haveUsedDiscoveryAction), f(bc.undefinedBehaviorEvoked), f(bc.loopCount), f(bc.energyWasted),
            f(bc.cardsDrawn), f(bc.hiddenInformationAccesses), f(bc.outcome), f(bc.unsupportedEffectKind),
            f(bc.unsupportedEffectId), f(bc.inputState, value(d.inputState)), f(bc.monsterTurnIdx, d.monsterTurnIdx),
            f(bc.isBattleOver), f(bc.escapedCombat), f(bc.endTurnQueued), f(bc.endTurnAfterCurrentCard),
            f(bc.turnHasEnded), f(bc.skipMonsterTurn), f(bc.potionCount), f(bc.potionCapacity, d.potionCapacity),
            f(bc.turn), f(misc_bits), f(bc.cardQueue.size), f(bc.cardQueue.frontIdx), f(bc.cardQueue.backIdx));
    if constexpr (C::reading) bc.miscBits = decltype(bc.miscBits)(misc_bits);
    for (auto& potion : bc.potions) c.u(potion);
    auto& select = bc.cardSelectInfo;
    c.group(f(select.cards[0]), f(select.cards[1]), f(select.cards[2]), f(select.canPickZero),
            f(select.canPickAnyNumber), f(select.pickCount), f(select.data0),
            f(select.cardSelectTask, value(d.cardSelectInfo.cardSelectTask)));

    auto& p = bc.player;
    const auto& dp = d.player;
    std::uint64_t pellets = p.orangePelletsCardTypesPlayed.to_ulong();
    c.group(f(p.cc, value(dp.cc)), f(p.gold), f(p.curHp, dp.curHp), f(p.maxHp, dp.maxHp), f(p.energy),
            f(p.energyPerTurn, dp.energyPerTurn), f(p.cardDrawPerTurn, dp.cardDrawPerTurn), f(p.stance),
            f(p.orbSlots), f(p.lightningOrbsChanneled), f(p.frostOrbsChanneled),
            f(p.lastTargetedMonster, dp.lastTargetedMonster), f(p.block), f(p.artifact), f(p.dexterity), f(p.focus),
            f(p.strength), f(p.justAppliedBits), f(p.statusBits0), f(p.statusBits1), f(p.relicBits0), f(p.relicBits1));
    c.group(f(p.happyFlowerCounter), f(p.incenseBurnerCounter), f(p.inkBottleCounter), f(p.inserterCounter),
            f(p.nunchakuCounter), f(p.penNibCounter), f(p.sundialCounter), f(p.haveUsedNecronomiconThisTurn),
            f(p.combustHpLoss), f(p.devaFormEnergyPerTurn), f(p.echoFormCardsDoubled), f(p.panacheCounter),
            f(p.cardsPlayedThisTurn), f(p.attacksPlayedThisTurn), f(p.skillsPlayedThisTurn), f(pellets),
            f(p.cardsDiscardedThisTurn), f(p.lastAttackUnblockedDamage), f(p.timesDamagedThisCombat), f(p.bomb1),
            f(p.bomb2), f(p.bomb3), f(p.selfRepairHeal), f(p.creativeAi), f(p.helloWorld), f(p.heatsinks),
            f(p.nirvanaBlock), f(p.rushdownDraw), f(p.studyInsights), f(p.mentalFortressBlock), f(p.machineLearning),
            f(p.storm));
    if constexpr (C::reading) p.orangePelletsCardTypesPlayed = decltype(p.orangePelletsCardTypesPlayed)(pellets);
    std::uint64_t orbs = 0;
    for (std::size_t i = 0; i < p.orbs.size(); ++i) orbs |= std::uint64_t(p.orbs[i] != Orb{} || p.orbData[i]) << i;
    c.varint(orbs);
    for (std::size_t i = 0; i < p.orbs.size(); ++i)
        if (orbs >> i & 1) {
            c.u(p.orbs[i]);
            c.s(p.orbData[i]);
        }
    std::uint64_t statuses = p.statusMap.size();
    c.varint(statuses);
    if constexpr (C::reading) {
        for (std::uint64_t i = 0; i < statuses; ++i) {
            PlayerStatus status{};
            std::int16_t amount = 0;
            c.u(status);
            c.s(amount);
            p.statusMap[status] = amount;
        }
    } else {
        for (auto [status, amount] : p.statusMap) {
            c.u(status);
            c.s(amount);
        }
    }

    auto& m = bc.monsters;
    std::uint64_t turn_bits = m.extraRollMoveOnTurn.to_ulong() | m.skipTurn.to_ulong() << 5;
    c.group(f(m.monsterCount), f(m.monstersAlive), f(turn_bits));
    if constexpr (C::reading) {
        m.extraRollMoveOnTurn = decltype(m.extraRollMoveOnTurn)(turn_bits & 0x1F);
        m.skipTurn = decltype(m.skipTurn)(turn_bits >> 5);
    }
    if (m.monsterCount < 0 || m.monsterCount > int(m.arr.size())) throw std::invalid_argument{"unpack: monster count"};
    for (int i = 0; i < m.monsterCount; ++i) {
        auto& x = m.arr[i];
        c.u(x.id);
        c.group(f(x.curHp), f(x.maxHp), f(x.block), f(x.isEscapingB), f(x.halfDead), f(x.escapeNext),
                f(x.moveHistory[0], value(MMID::INVALID)), f(x.moveHistory[1], value(MMID::INVALID)), f(x.statusBits),
                f(x.artifact), f(x.blockReturn), f(x.choked), f(x.corpseExplosion), f(x.lockOn), f(x.mark),
                f(x.metallicize), f(x.platedArmor), f(x.poison), f(x.regen), f(x.shackled), f(x.strength),
                f(x.vulnerable), f(x.weak), f(x.uniquePower0), f(x.uniquePower1), f(x.miscInfo), f(x.idx, i));
    }

    auto& k = bc.cards;
    c.group(f(k.nextUniqueCardId), f(k.cardsInHand), f(k.handNormalityCount), f(k.handPainCount), f(k.strikeCount),
            f(k.handBloodCardCount), f(k.drawPileBloodCardCount), f(k.discardPileBloodCardCount));
    if (k.cardsInHand < 0 || k.cardsInHand > int(k.hand.size())) throw std::invalid_argument{"unpack: hand size"};
    for (int i = 0; i < k.cardsInHand; ++i) card(c, k.hand[i]);
    card_slots(c, k.limbo);
    card_slots(c, k.stasisCards);
    cards(c, k.drawPile);
    cards(c, k.discardPile);
    cards(c, k.exhaustPile);

    auto& q = bc.cardQueue;
    if (q.size < 0 || q.size > int(q.arr.size())) throw std::invalid_argument{"unpack: card queue size"};
    for (int i = 0; i < q.size; ++i) queue_item(c, q.arr[(q.frontIdx + i) % q.arr.size()]);
    queue_item(c, bc.curCardQueueItem);
    std::uint64_t nightmares = bc.nightmareCards.size();
    c.varint(nightmares);
    if constexpr (C::reading) {
        if (nightmares > 10) throw std::invalid_argument{"unpack: nightmare count"};
        bc.nightmareCards.resize(int(nightmares));
        bc.nightmareCopyCounts.resize(int(nightmares));
    }
    for (int i = 0; i < int(nightmares); ++i) {
        card(c, bc.nightmareCards[i]);
        c.s(bc.nightmareCopyCounts[i]);
    }
}
}  // namespace

bool packable(const BattleContext& state) { return state.actionQueue.size == 0; }

void pack(const BattleContext& state, std::string& out) {
    if (!packable(state)) throw std::invalid_argument{"pack: queued actions cannot be stored"};
    Writer writer{out};
    code(writer, const_cast<BattleContext&>(state));  // a Writer only reads the state
}

BattleContext unpack(std::span<const char> bytes) {
    BattleContext state;
    Reader reader{bytes.data(), bytes.data() + bytes.size()};
    code(reader, state);
    if (reader.p != reader.end) throw std::invalid_argument{"unpack: trailing bytes"};
    return state;
}
}  // namespace stsrl::combat
