#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include "environments/combat/state_store.hpp"
#include "constants/CardPools.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <future>
#include <numeric>
#include <random>

namespace stsrl::combat {
namespace {
// Random play through fights across the acts, with random decks, so that many kinds of state come up.
const std::vector<sts::BattleContext>& states() {
    static const auto all = [] {
        using E = sts::MonsterEncounter;
        const std::pair<E, int> encounters[] = {
            {E::JAW_WORM, 1},     {E::GREMLIN_NOB, 1}, {E::THREE_SENTRIES, 1},  {E::SLIME_BOSS, 1},
            {E::GREMLIN_LEADER, 2}, {E::CHAMP, 2},     {E::AUTOMATON, 2},       {E::COLLECTOR, 2},
            {E::REPTOMANCER, 3},  {E::AWAKENED_ONE, 3}, {E::TIME_EATER, 3},     {E::DONU_AND_DECA, 3}};
        std::mt19937_64 random{2024};
        std::vector<sts::BattleContext> out;
        for (int fight = 0; fight < 150; ++fight) {
            const auto [encounter, act] = encounters[fight % std::size(encounters)];
            sts::GameContext game{sts::CharacterClass::IRONCLAD, random(), 20};
            game.act = act;
            game.floorNum = act * 17 - 8;
            for (int i = 0; i < 15; ++i) {
                sts::Card card{sts::CombatCardPool::getCardAt(sts::CharacterClass::IRONCLAD, int(random() % 70))};
                card.upgraded = random() % 2;
                game.deck.obtainRaw(card);
            }
            sts::BattleContext state;
            state.init(game, encounter);
            while (state.outcome == sts::Outcome::UNDECIDED && state.turn < 30) {
                const auto actions = legal_actions(state);
                const auto action = actions[random() % actions.size()];
                out.push_back(state);
                action.execute(state);
            }
        }
        return out;
    }();
    return all;
}

std::string packed(const sts::BattleContext& state) {
    std::string out;
    pack(state, out);
    return out;
}

// Two states are alike when everything a player sees and every move's outcome agree, to the end of the fight.
void expect_plays_alike(sts::BattleContext a, sts::BattleContext b, std::uint64_t seed) {
    std::mt19937_64 random{seed};
    while (a.outcome == sts::Outcome::UNDECIDED && a.turn < 40) {
        ASSERT_EQ(observation_key(a), observation_key(b));
        if (packable(a)) {
            ASSERT_EQ(packed(a), packed(b));
        }
        const auto actions = legal_actions(a), others = legal_actions(b);
        ASSERT_EQ(actions.size(), others.size());
        for (std::size_t i = 0; i < actions.size(); ++i) ASSERT_EQ(action_key(a, actions[i]), action_key(b, others[i]));
        const auto i = random() % actions.size();
        actions[i].execute(a);
        others[i].execute(b);
    }
    EXPECT_EQ(a.outcome, b.outcome);
    EXPECT_EQ(a.player.curHp, b.player.curHp);
}

TEST(StateCodec, UnpackedStatePlaysOnLikeTheOriginal) {
    int checked = 0;
    for (const auto& state : states()) {
        if (!packable(state)) continue;
        const auto bytes = packed(state);
        const auto copy = unpack(bytes);
        ASSERT_EQ(packed(copy), bytes);
        expect_plays_alike(state, copy, checked++);
    }
    EXPECT_GT(checked, 1000);
}

TEST(StateCodec, QueuedActionsCannotBePacked) {
    const auto queued = std::ranges::find_if(states(), [](const auto& state) { return !packable(state); });
    ASSERT_NE(queued, states().end());
    EXPECT_FALSE(packable(*queued));
    std::string out;
    EXPECT_THROW(pack(*queued, out), std::invalid_argument);
}

TEST(StateCodec, RejectsTruncatedAndTrailingBytes) {
    const auto bytes = packed(states().front());
    for (std::size_t n = 0; n < bytes.size(); ++n)
        EXPECT_THROW(unpack({bytes.data(), n}), std::invalid_argument) << n << " of " << bytes.size() << " bytes";
    EXPECT_THROW(unpack(bytes + '\0'), std::invalid_argument);
}

std::filesystem::path store_path(const char* name) { return std::filesystem::path{testing::TempDir()} / name; }

void write_all(const std::filesystem::path& path) {
    StateWriter writer{path};
    for (const auto& state : states()) {
        if (!packable(state)) continue;
        const auto next = writer.size();
        EXPECT_EQ(writer.append(state), next);
    }
    writer.flush();
}

TEST(StateStore, ReadsEveryRecordInAnyOrderFromSeveralThreads) {
    std::vector<sts::BattleContext> all;
    std::ranges::copy_if(states(), std::back_inserter(all), packable);
    const auto path = store_path("states.bin");
    write_all(path);

    const StateReader reader{path};
    ASSERT_EQ(reader.size(), all.size());
    std::vector<std::size_t> order(all.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::shuffle(order, std::mt19937_64{7});
    std::vector<std::future<void>> readers;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        readers.push_back(std::async(std::launch::async, [&, lane] {
            for (std::size_t j = lane; j < order.size(); j += 4) {
                const auto i = order[j];
                ASSERT_EQ(packed(reader.at(i)), packed(all[i])) << "record " << i;
            }
        }));
    }
    for (auto& task : readers) task.get();
}

TEST(StateStore, FlushMakesAppendedRecordsReadable) {
    const auto path = store_path("flushed.bin");
    StateWriter writer{path};
    writer.flush();
    EXPECT_EQ(StateReader{path}.size(), 0);
    EXPECT_EQ(writer.append(states().front()), 0);
    writer.flush();
    const StateReader reader{path};
    EXPECT_EQ(reader.size(), writer.size());
    EXPECT_EQ(packed(reader.at(0)), packed(states().front()));
    EXPECT_THROW(reader.at(reader.size()), std::out_of_range);
}

TEST(StateStore, RejectedAppendLeavesTheStoreUsable) {
    const auto& all = states();
    const auto queued = std::ranges::find_if(all, [](const auto& state) { return !packable(state); });
    ASSERT_NE(queued, all.end());
    StateWriter writer{store_path("queued.bin")};
    EXPECT_THROW(writer.append(*queued), std::invalid_argument);
    EXPECT_EQ(writer.size(), 0);
    const auto& state = states().front();
    EXPECT_EQ(writer.append(state), 0);
    writer.flush();
    const StateReader reader{store_path("queued.bin")};
    ASSERT_EQ(reader.size(), 1);
    EXPECT_EQ(packed(reader.at(0)), packed(state));
}

TEST(StateStore, TruncatedFileIsRejected) {
    const auto path = store_path("truncated.bin");
    {
        StateWriter writer{path};
        writer.append(states().front());
    }
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    EXPECT_THROW(StateReader{path}, std::runtime_error);
}
}  // namespace
}  // namespace stsrl::combat
