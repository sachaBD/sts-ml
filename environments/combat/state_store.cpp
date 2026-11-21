#include "environments/combat/state_store.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace stsrl::combat {
namespace {
// File: magic and version, then length-prefixed records (kind byte and payload).
// All scalars use native representation, like the state codec.
constexpr std::string_view magic = "STSSTATE";
constexpr std::uint32_t version = 3;
enum Kind : char { packed = 0, after_predecessor = 1 };

template <class T> void put(std::string& out, T value) { out.append(reinterpret_cast<const char*>(&value), sizeof value); }
template <class T> T get(const char*& p, const char* end) {
    if (std::size_t(end - p) < sizeof(T)) throw std::runtime_error{"state store: truncated"};
    T value;
    std::memcpy(&value, p, sizeof value);
    p += sizeof value;
    return value;
}

}  // namespace

StateWriter::StateWriter(const std::filesystem::path& path) : out_{path, std::ios::binary | std::ios::trunc} {
    if (!out_) throw std::runtime_error{"state store: cannot write " + path.string()};
    std::string header{magic};
    put(header, version);
    out_.write(header.data(), std::streamsize(header.size()));
    if (!out_) throw std::runtime_error{"state store: write failed"};
}

std::uint64_t StateWriter::append(const sts::BattleContext& state, std::optional<Predecessor> predecessor) {
    record_.assign(sizeof(std::uint32_t), '\0');  // its size, filled in below
    if (packable(state)) {
        record_.push_back(Kind::packed);
        pack(state, record_);
    } else {
        if (!predecessor) throw std::invalid_argument{"state store: a state with queued actions needs its predecessor"};
        if (predecessor->record >= records_) throw std::invalid_argument{"state store: predecessor not yet stored"};
        record_.push_back(Kind::after_predecessor);
        put(record_, predecessor->record);
        put(record_, predecessor->action.bits);
    }
    const auto size = static_cast<std::uint32_t>(record_.size() - sizeof(std::uint32_t));
    std::memcpy(record_.data(), &size, sizeof size);
    out_.write(record_.data(), std::streamsize(record_.size()));
    if (!out_) throw std::runtime_error{"state store: write failed"};
    return records_++;
}

void StateWriter::flush() {
    out_.flush();
    if (!out_) throw std::runtime_error{"state store: write failed"};
}

StateReader::StateReader(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) throw std::runtime_error{"state store: cannot read " + path.string()};
    bytes_.assign(std::istreambuf_iterator<char>{in}, {});
    const char* p = bytes_.data();
    const char* end = p + bytes_.size();
    if (bytes_.size() < magic.size() || std::string_view{p, magic.size()} != magic)
        throw std::runtime_error{"state store: not a state store: " + path.string()};
    p += magic.size();
    if (const auto found = get<std::uint32_t>(p, end); found != version)
        throw std::runtime_error{"state store: unsupported version " + std::to_string(found)};
    while (p != end) {
        const auto size = get<std::uint32_t>(p, end);
        if (size == 0 || std::size_t(end - p) < size) throw std::runtime_error{"state store: truncated record"};
        records_.push_back({std::uint64_t(p - bytes_.data()), size});
        p += size;
    }
}

sts::BattleContext StateReader::at(std::uint64_t record) const {
    if (record >= size()) throw std::out_of_range{"state store: no record " + std::to_string(record)};
    const auto [offset, length] = records_[record];
    const char* p = bytes_.data() + offset;
    const char* end = p + length;
    const char kind = *p++;
    if (kind == Kind::after_predecessor) {
        const auto previous = get<std::uint64_t>(p, end);
        const sts::search::Action action{get<std::uint32_t>(p, end)};
        if (previous >= record) throw std::runtime_error{"state store: predecessor after its record"};
        auto state = at(previous);
        action.execute(state);
        play_forced(state);
        return state;
    }
    if (kind != Kind::packed) throw std::runtime_error{"state store: unknown record kind"};
    return unpack({p, std::size_t(end - p)});
}
}  // namespace stsrl::combat
