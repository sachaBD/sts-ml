#include "environments/combat/state_store.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include <zdict.h>
#include <zstd.h>
#include <cstring>
#include <stdexcept>

namespace stsrl::combat {
namespace {
// File: magic, version, dictionary size and dictionary, then records.
// Record: its size, then a kind byte and its payload. Integers are little-endian, as the machine writes them.
constexpr std::string_view magic = "STSSTATE";
constexpr std::uint32_t version = 2;
constexpr int level = 3;
enum Kind : char { packed = 0, after_predecessor = 1 };

template <class T> void put(std::string& out, T value) { out.append(reinterpret_cast<const char*>(&value), sizeof value); }
template <class T> T get(const char*& p, const char* end) {
    if (std::size_t(end - p) < sizeof(T)) throw std::runtime_error{"state store: truncated"};
    T value;
    std::memcpy(&value, p, sizeof value);
    p += sizeof value;
    return value;
}

void check(std::size_t code, const char* what) {
    if (ZSTD_isError(code)) throw std::runtime_error{std::string{what} + ": " + ZSTD_getErrorName(code)};
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
    std::string header{magic};
    put(header, version);
    put(header, static_cast<std::uint32_t>(dictionary.size()));
    header += dictionary;
    out_.write(header.data(), std::streamsize(header.size()));
}

StateWriter::~StateWriter() = default;

std::uint64_t StateWriter::append(const sts::BattleContext& state, std::optional<Predecessor> predecessor) {
    record_.assign(sizeof(std::uint32_t), '\0');  // its size, filled in below
    if (packable(state)) {
        packed_.clear();
        pack(state, packed_);
        record_.push_back(Kind::packed);
        const auto at = record_.size();
        record_.resize(at + ZSTD_compressBound(packed_.size()));
        const auto size = ZSTD_compress2(zstd_->context, record_.data() + at, record_.size() - at, packed_.data(),
                                         packed_.size());
        check(size, "zstd compress");
        record_.resize(at + size);
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
    if (bytes_.size() < magic.size() || std::string_view{p, magic.size()} != magic)
        throw std::runtime_error{"state store: not a state store: " + path.string()};
    p += magic.size();
    if (const auto found = get<std::uint32_t>(p, end); found != version)
        throw std::runtime_error{"state store: unsupported version " + std::to_string(found)};
    const auto dictionary_size = get<std::uint32_t>(p, end);
    if (std::size_t(end - p) < dictionary_size) throw std::runtime_error{"state store: truncated dictionary"};
    if (dictionary_size) {
        zstd_->dictionary = ZSTD_createDDict(p, dictionary_size);
        if (!zstd_->dictionary) throw std::runtime_error{"state store: bad dictionary"};
    }
    p += dictionary_size;
    while (p != end) {
        const auto size = get<std::uint32_t>(p, end);
        if (size == 0 || std::size_t(end - p) < size) throw std::runtime_error{"state store: truncated record"};
        records_.push_back({std::uint64_t(p - bytes_.data()), size});
        p += size;
    }
}

StateReader::~StateReader() = default;

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
