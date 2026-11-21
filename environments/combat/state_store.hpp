#pragma once
// Random-access combat states in an uncompressed, append-only file.
// Trusted, same-build data only; simulator field changes require regenerating the store.
// States with queued closures cannot be stored.
#include "combat/BattleContext.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace stsrl::combat {
class StateWriter {
public:
    // Creates (or truncates) the file. Single writer; flush before opening a reader.
    explicit StateWriter(const std::filesystem::path&);
    // Returns the record's index. Throws if the state has queued closures.
    std::uint64_t append(const sts::BattleContext&);
    std::uint64_t size() const { return records_; }
    void flush();

private:
    std::ofstream out_;
    std::uint64_t records_ = 0;
    std::string record_;
};

class StateReader {
public:
    explicit StateReader(const std::filesystem::path&);  // throws on a truncated or foreign file
    std::uint64_t size() const { return records_.size(); }
    sts::BattleContext at(std::uint64_t record) const;  // safe to call from several threads

private:
    std::string bytes_;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> records_;  // each record's offset and size in bytes_
};
}  // namespace stsrl::combat
