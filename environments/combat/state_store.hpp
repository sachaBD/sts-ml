#pragma once
// Fight states for random access: one append-only file, each record compressed on its own (zstd) with a
// dictionary the file carries, so any record decodes without the others.
// A state that cannot be packed (queued actions) is stored as the decision before it and the move played there.
#include "combat/BattleContext.h"
#include "sim/search/Action.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace stsrl::combat {
// A zstd dictionary for a store, trained on the packable ones of `states`. Throws if they are too few to train.
std::string train_dictionary(std::span<const sts::BattleContext> states, std::size_t capacity = 64 * 1024);

// The previous decision's record and the move played there; with forced moves after it, it reaches the state.
struct Predecessor {
    std::uint64_t record;
    sts::search::Action action;
};

class StateWriter {
public:
    // Creates (or truncates) the file. An empty dictionary compresses each record without one.
    StateWriter(const std::filesystem::path&, std::string_view dictionary);
    ~StateWriter();
    // Returns the record's index. A state that is not packable needs its predecessor.
    std::uint64_t append(const sts::BattleContext&, std::optional<Predecessor> = std::nullopt);
    std::uint64_t size() const { return records_; }
    void flush();

private:
    struct Zstd;
    std::unique_ptr<Zstd> zstd_;
    std::ofstream out_;
    std::uint64_t records_ = 0;
    std::string packed_, record_;
};

class StateReader {
public:
    explicit StateReader(const std::filesystem::path&);  // throws on a truncated or foreign file
    ~StateReader();
    std::uint64_t size() const { return records_.size(); }
    sts::BattleContext at(std::uint64_t record) const;  // safe to call from several threads

private:
    struct Zstd;
    std::unique_ptr<Zstd> zstd_;
    std::string bytes_;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> records_;  // each record's offset and size in bytes_
};
}  // namespace stsrl::combat
