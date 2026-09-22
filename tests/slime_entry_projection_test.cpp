#include "scenarios/slime_entry_projection.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <set>

int main() {
    const char* path = "slime-entry-projection-test.jsonl";
    std::ofstream out{path};
    out << R"({"status":"non_slime"})" << '\n';
    out << R"({"status":"accepted","seed":17,"act":1,"boss":{"name":"SLIME_BOSS"},"encounter":{"name":"SLIME_BOSS"},"hp":41,"max_hp":73,"deck_signature":"deck-hash","public_snapshot_sha256":"0123456789abcdef0123456789abcdef","deck":[{"id":321,"upgraded":0,"misc":0},{"id":25,"upgraded":1,"misc":7}]})" << '\n';
    out.close();
    int skipped = 0;
    const auto entries = stsrl::scenarios::load_slime_entry_projections(path, skipped);
    assert(skipped == 1 && entries.size() == 1);
    const auto& entry = entries.front();
    assert(entry.entry_id == "slime-entry-v1-17-0123456789abcdef");
    assert(entry.deck_signature == "deck-hash" && entry.public_snapshot_sha256.size() == 32);
    auto environment = stsrl::scenarios::slime_entry_projection(entry);
    assert(environment.player_hp() == 41 && environment.player_max_hp() == 73);
    const auto cards = environment.decision().encoding.cards;
    assert(cards.size() == 2);
    assert(std::count_if(cards.begin(), cards.end(), [](const auto& card) { return card.card_id == 25 && card.numeric[0] == 1.f && card.numeric[8] == .7f; }) == 1);
    std::remove(path);
}
