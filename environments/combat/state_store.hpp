#pragma once
// Only unmodified output from the same simulator build is supported. Pending action closures cannot be saved.
#include "combat/BattleContext.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
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
    explicit StateReader(const std::filesystem::path&);
    std::uint64_t size() const { return records_.size(); }
    sts::BattleContext at(std::uint64_t record) const;  // safe to call from several threads

private:
    struct Record {
        std::uint64_t offset;
        std::uint32_t size;
    };
    std::string bytes_;
    std::vector<Record> records_;
};
}  // namespace stsrl::combat
