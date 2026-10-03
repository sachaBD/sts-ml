// combat_v4_full (runs/schema=combat_v4_full/schema.py): expand combat_v4 fights into Arrow tables, one row per
// decision (`decisions`) and one per fight (`fights`). Arrow only, no intermediate text format.
//
//   Expander::add(fight)   rebuild the fight like combat_v4::replay, snapshot the state before every action.
//
// add() returns false (reason filled, nothing appended) when the recording does not replay: invalid action, ended
// early / not at the end, chosen move not enumerated, won / final_hp differ. Anything else is a contract violation and
// throws std::logic_error: non-Ironclad state, undecoded monster raw integer, empty legal set.
#pragma once

#include "environments/combat/record_v4.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>

namespace stsrl::combat_v4_full {

struct SearchChild { std::uint32_t action, visits; float value; };
struct SearchStep { float root_value; std::uint32_t simulations; std::vector<SearchChild> children; };

struct Fight {
    std::string fight_id, agent;
    combat_v4::Start start;
    std::vector<std::uint32_t> actions;
    std::vector<char> explored;                  // aligned with actions
    bool won;
    int final_hp;
    std::vector<const SearchStep*> search;       // aligned with actions; nullptr: no search row
};

// The two tables' Arrow schemas, identical to runs/schema=combat_v4_full/schema.py (checked by the unit test).
std::shared_ptr<arrow::Schema> decisions_schema();
std::shared_ptr<arrow::Schema> fights_schema();

class Expander {
public:
    Expander();
    ~Expander();
    bool add(const Fight& fight, std::string& mismatch);
    std::int64_t decisions() const;
    std::int64_t fights() const;
    // finish the tables built so far (resets the builders)
    std::shared_ptr<arrow::Table> take_decisions();
    std::shared_ptr<arrow::Table> take_fights();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stsrl::combat_v4_full
