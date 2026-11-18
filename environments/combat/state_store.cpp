#include "environments/combat/state_store.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include <zdict.h>
#include <zstd.h>
#include <cstring>
#include <stdexcept>

namespace stsrl::combat {
namespace {
// File: magic, version, dictionary size (u32 little-endian) and dictionary, then records.
// Record: its size as a varint, then a kind byte and its payload.
constexpr std::string_view magic = "STSSTATE";
constexpr std::uint32_t version = 1;
constexpr int level = 3;
enum Kind : char { packed = 0, after_predecessor = 1 };

void put_varint(std::string& out, std::uint64_t v) {
    for (; v >= 0x80; v >>= 7) out.push_back(char(v | 0x80));
    out.push_back(char(v));
}
std::uint64_t get_varint(const char*& p, const char* end) {
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (p == end) throw std::runtime_error{"state store: truncated record"};
        const auto byte = static_cast<unsigned char>(*p++);
        v |= std::uint64_t(byte & 0x7F) << shift;
        if (!(byte & 0x80)) return v;
    }
    throw std::runtime_error{"state store: bad varint"};
}

void check(std::size_t code, const char* what) {
    if (ZSTD_isError(code)) throw std::runtime_error{std::string{what} + ": " + ZSTD_getErrorName(code)};
}

// Plays on from a decision to the next state with a choice, as decisions are recorded.
void play(sts::BattleContext& state, sts::search::Action action) {
    action.execute(state);
    while (state.outcome == sts::Outcome::UNDECIDED) {
        const auto actions = legal_actions(state);
        if (actions.size() != 1) return;
        actions[0].execute(state);
    }
}
}  // namespace

std::string train_dictionary(std::span<const sts::BattleContext> states, std::size_t capacity) {
    std::string samples;
    std::vector<std::size_t> sizes;
    for (const auto& state : states) {
        if (!packable(state)) continue;
        const auto before = samples.size();
        pack(state, samples);
        sizes.push_back(samples.size() - before);
    }
    std::string dictionary(capacity, '\0');
    const auto size = ZDICT_trainFromBuffer(dictionary.data(), dictionary.size(), samples.data(), sizes.data(),
                                            unsigned(sizes.size()));
    if (ZDICT_isError(size)) throw std::invalid_argument{std::string{"train_dictionary: "} + ZDICT_getErrorName(size)};
    dictionary.resize(size);
    return dictionary;
}

struct StateWriter::Zstd {
    ZSTD_CCtx* context = ZSTD_createCCtx();
    ZSTD_CDict* dictionary = nullptr;
    ~Zstd() {
        ZSTD_freeCDict(dictionary);
        ZSTD_freeCCtx(context);
    }
};

StateWriter::StateWriter(const std::filesystem::path& path, std::string_view dictionary)
    : zstd_{std::make_unique<Zstd>()}, out_{path, std::ios::binary | std::ios::trunc} {
    if (!out_) throw std::runtime_error{"state store: cannot write " + path.string()};
    check(ZSTD_CCtx_setParameter(zstd_->context, ZSTD_c_compressionLevel, level), "zstd level");
    check(ZSTD_CCtx_setParameter(zstd_->context, ZSTD_c_dictIDFlag, 0), "zstd dictionary id");  // one per file
    if (!dictionary.empty()) {
        zstd_->dictionary = ZSTD_createCDict(dictionary.data(), dictionary.size(), level);
        if (!zstd_->dictionary) throw std::invalid_argument{"state store: bad dictionary"};
        check(ZSTD_CCtx_refCDict(zstd_->context, zstd_->dictionary), "zstd dictionary");
    }
    const auto dictionary_size = static_cast<std::uint32_t>(dictionary.size());
    out_.write(magic.data(), magic.size());
    out_.write(reinterpret_cast<const char*>(&version), sizeof version);
    out_.write(reinterpret_cast<const char*>(&dictionary_size), sizeof dictionary_size);
    out_.write(dictionary.data(), std::streamsize(dictionary.size()));
}

StateWriter::~StateWriter() = default;

std::uint64_t StateWriter::append(const sts::BattleContext& state, std::optional<Predecessor> predecessor) {
    record_.clear();
    if (packable(state)) {
        packed_.clear();
        pack(state, packed_);
        std::string frame(ZSTD_compressBound(packed_.size()), '\0');
        const auto size = ZSTD_compress2(zstd_->context, frame.data(), frame.size(), packed_.data(), packed_.size());
        check(size, "zstd compress");
        put_varint(record_, 1 + size);
        record_.push_back(Kind::packed);
        record_.append(frame.data(), size);
    } else {
        if (!predecessor) throw std::invalid_argument{"state store: a state with queued actions needs its predecessor"};
        if (predecessor->record >= records_) throw std::invalid_argument{"state store: predecessor not yet stored"};
        std::string payload;
        put_varint(payload, predecessor->record);
        put_varint(payload, predecessor->action.bits);
        put_varint(record_, 1 + payload.size());
        record_.push_back(Kind::after_predecessor);
        record_ += payload;
    }
    out_.write(record_.data(), std::streamsize(record_.size()));
    if (!out_) throw std::runtime_error{"state store: write failed"};
    return records_++;
}

void StateWriter::flush() {
    out_.flush();
    if (!out_) throw std::runtime_error{"state store: write failed"};
}

struct StateReader::Zstd {
    ZSTD_DDict* dictionary = nullptr;
    ~Zstd() { ZSTD_freeDDict(dictionary); }
};

StateReader::StateReader(const std::filesystem::path& path) : zstd_{std::make_unique<Zstd>()} {
    std::ifstream in{path, std::ios::binary};
    if (!in) throw std::runtime_error{"state store: cannot read " + path.string()};
    bytes_.assign(std::istreambuf_iterator<char>{in}, {});
    const char* p = bytes_.data();
    const char* end = p + bytes_.size();
    std::uint32_t file_version = 0, dictionary_size = 0;
    const auto header = magic.size() + sizeof file_version + sizeof dictionary_size;
    if (bytes_.size() < header || std::string_view{p, magic.size()} != magic)
        throw std::runtime_error{"state store: not a state store: " + path.string()};
    std::memcpy(&file_version, p + magic.size(), sizeof file_version);
    std::memcpy(&dictionary_size, p + magic.size() + sizeof file_version, sizeof dictionary_size);
    if (file_version != version) throw std::runtime_error{"state store: unsupported version " + std::to_string(file_version)};
    p += header;
    if (std::size_t(end - p) < dictionary_size) throw std::runtime_error{"state store: truncated dictionary"};
    if (dictionary_size) {
        zstd_->dictionary = ZSTD_createDDict(p, dictionary_size);
        if (!zstd_->dictionary) throw std::runtime_error{"state store: bad dictionary"};
    }
    p += dictionary_size;
    while (p != end) {
        offsets_.push_back(std::uint64_t(p - bytes_.data()));
        const auto size = get_varint(p, end);
        if (std::uint64_t(end - p) < size || size == 0) throw std::runtime_error{"state store: truncated record"};
        p += size;
    }
}

StateReader::~StateReader() = default;

sts::BattleContext StateReader::at(std::uint64_t record) const {
    if (record >= size()) throw std::out_of_range{"state store: no record " + std::to_string(record)};
    const char* p = bytes_.data() + offsets_[record];
    const auto length = get_varint(p, bytes_.data() + bytes_.size());
    const char* end = p + length;
    const char kind = *p++;
    if (kind == Kind::after_predecessor) {
        const auto previous = get_varint(p, end);
        const sts::search::Action action{static_cast<std::uint32_t>(get_varint(p, end))};
        if (previous >= record) throw std::runtime_error{"state store: predecessor after its record"};
        auto state = at(previous);
        play(state, action);
        return state;
    }
    if (kind != Kind::packed) throw std::runtime_error{"state store: unknown record kind"};
    // A decompression context per thread: contexts are not shareable, dictionaries are.
    thread_local const std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context{ZSTD_createDCtx(), ZSTD_freeDCtx};
    const auto size = ZSTD_getFrameContentSize(p, std::size_t(end - p));
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN)
        throw std::runtime_error{"state store: bad frame"};
    std::string packed(size, '\0');
    const auto got = zstd_->dictionary
                         ? ZSTD_decompress_usingDDict(context.get(), packed.data(), packed.size(), p,
                                                      std::size_t(end - p), zstd_->dictionary)
                         : ZSTD_decompressDCtx(context.get(), packed.data(), packed.size(), p, std::size_t(end - p));
    check(got, "zstd decompress");
    return unpack({packed.data(), got});
}
}  // namespace stsrl::combat
