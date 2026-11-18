#include "environments/combat/state_codec.hpp"
#include "combat/StateFields.h"
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace stsrl::combat {
namespace {
// Every leaf of the simulator's walk (sts::forEachField) is a zigzag varint, except the RNGs.
void put(std::string& out, std::uint64_t v) {
    for (; v >= 0x80; v >>= 7) out.push_back(char(v | 0x80));
    out.push_back(char(v));
}

std::uint64_t get(const char*& p, const char* end) {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (p == end) throw std::invalid_argument{"unpack: truncated state"};
        const auto byte = static_cast<unsigned char>(*p++);
        v |= std::uint64_t(byte & 0x7F) << shift;
        if (!(byte & 0x80)) return v;
    }
    throw std::invalid_argument{"unpack: bad varint"};
}

std::uint64_t zigzag(std::int64_t v) { return (std::uint64_t(v) << 1) ^ std::uint64_t(v >> 63); }
std::int64_t unzigzag(std::uint64_t z) { return std::int64_t(z >> 1) ^ -std::int64_t(z & 1); }

// The fight's RNGs begin at Random(seed + floor) (BattleContext::init; the walk visits seed and floor first), so
// each is written as its number of steps from there: about two bytes instead of sixteen random ones. The run's RNGs
// (misc, potion) start elsewhere, and any RNG not reached within the limit, are written whole.
constexpr std::uint64_t step_limit = 1 << 16;
bool from_fight_seed(const sts::BattleContext& state, const sts::Random& rng) {
    return &rng != &state.miscRng && &rng != &state.potionRng;
}
sts::Random fight_start(const sts::BattleContext& state) { return sts::Random{state.seed + state.floorNum}; }
constexpr std::uint64_t whole = 1;  // the low bit of an RNG's tag; otherwise the tag is its steps, shifted
}  // namespace

bool packable(const sts::BattleContext& state) { return state.actionQueue.size == 0; }

void pack(const sts::BattleContext& state, std::string& out) {
    if (!packable(state)) throw std::invalid_argument{"pack: queued actions cannot be stored"};
    sts::forEachField(state, [&](const auto& leaf) {
        if constexpr (std::is_same_v<std::decay_t<decltype(leaf)>, sts::Random>) {
            std::uint64_t tag = whole;
            if (from_fight_seed(state, leaf)) {
                auto walk = fight_start(state);
                for (std::uint64_t n = 0; n <= step_limit; ++n, walk.nextLong())
                    if (walk.seed0 == leaf.seed0 && walk.seed1 == leaf.seed1) {
                        tag = n << 1;
                        break;
                    }
            }
            put(out, tag);
            if (tag == whole) {
                out.append(reinterpret_cast<const char*>(&leaf.seed0), sizeof leaf.seed0);
                out.append(reinterpret_cast<const char*>(&leaf.seed1), sizeof leaf.seed1);
            }
            put(out, zigzag(leaf.counter));
        } else {
            put(out, zigzag(static_cast<std::int64_t>(leaf)));
        }
    });
}

sts::BattleContext unpack(std::span<const char> bytes) {
    sts::BattleContext state;
    const char* p = bytes.data();
    const char* end = p + bytes.size();
    sts::forEachField(state, [&](auto& leaf) {
        using Leaf = std::decay_t<decltype(leaf)>;
        if constexpr (std::is_same_v<Leaf, sts::Random>) {
            const auto tag = get(p, end);
            if (tag == whole) {
                if (end - p < std::ptrdiff_t(sizeof leaf.seed0 + sizeof leaf.seed1))
                    throw std::invalid_argument{"unpack: truncated state"};
                std::memcpy(&leaf.seed0, p, sizeof leaf.seed0);
                std::memcpy(&leaf.seed1, p + sizeof leaf.seed0, sizeof leaf.seed1);
                p += sizeof leaf.seed0 + sizeof leaf.seed1;
            } else {
                if (tag & whole || (tag >> 1) > step_limit) throw std::invalid_argument{"unpack: bad RNG"};
                leaf = fight_start(state);
                for (std::uint64_t n = 0; n < tag >> 1; ++n) leaf.nextLong();
            }
            leaf.counter = static_cast<std::int32_t>(unzigzag(get(p, end)));
        } else {
            leaf = static_cast<Leaf>(unzigzag(get(p, end)));
        }
    });
    if (p != end) throw std::invalid_argument{"unpack: trailing bytes"};
    return state;
}
}  // namespace stsrl::combat
