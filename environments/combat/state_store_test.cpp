#include "environments/combat/public.hpp"
#include "environments/combat/state_codec.hpp"
#include "environments/combat/state_store.hpp"
#include "constants/CardPools.h"
#include "game/GameContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unistd.h>

namespace stsrl::combat {
namespace {
// Random play through fights across the acts, with random decks, so that many kinds of state come up.
const std::vector<sts::BattleContext>& states() {
    static const auto all = [] {
        using E = sts::MonsterEncounter;
        constexpr auto character = sts::CharacterClass::IRONCLAD;
        constexpr auto card_count = sts::CombatCardPool::getPoolSize(character);
        const std::pair<E, int> encounters[] = {
            {E::JAW_WORM, 1},       {E::GREMLIN_NOB, 1},  {E::THREE_SENTRIES, 1}, {E::SLIME_BOSS, 1},
            {E::GREMLIN_LEADER, 2}, {E::CHAMP, 2},        {E::AUTOMATON, 2},      {E::COLLECTOR, 2},
            {E::REPTOMANCER, 3},    {E::AWAKENED_ONE, 3}, {E::TIME_EATER, 3},     {E::DONU_AND_DECA, 3}};
        std::mt19937_64 random{2024};
        std::vector<sts::BattleContext> out;
        for (int fight = 0; fight < 150; ++fight) {
            const auto [encounter, act] = encounters[fight % std::size(encounters)];
            sts::GameContext game{character, random(), 20};
            game.act = act;
            game.floorNum = act * 17 - 8;
            for (int i = 0; i < 15; ++i) {
                sts::Card card{sts::CombatCardPool::getCardAt(character, int(random() % card_count))};
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

// Check behavior independently of the codec's own representation.
void expect_plays_alike(sts::BattleContext a, sts::BattleContext b, std::uint64_t seed) {
    std::mt19937_64 random{seed};
    while (a.outcome == sts::Outcome::UNDECIDED && a.turn < 40) {
        ASSERT_EQ(a.outcome, b.outcome);
        ASSERT_EQ(a.turn, b.turn);
        ASSERT_EQ(observation_key(a), observation_key(b));
        ASSERT_EQ(packed(a), packed(b));
        const auto actions = legal_actions(a), others = legal_actions(b);
        ASSERT_EQ(actions.size(), others.size());
        for (std::size_t i = 0; i < actions.size(); ++i) ASSERT_EQ(action_key(a, actions[i]), action_key(b, others[i]));
        const auto i = random() % actions.size();
        actions[i].execute(a);
        others[i].execute(b);
    }
    EXPECT_EQ(a.turn, b.turn);
    EXPECT_EQ(a.outcome, b.outcome);
    EXPECT_EQ(a.player.curHp, b.player.curHp);
}

TEST(StateCodec, UnpackedStatePlaysOnLikeTheOriginal) {
    int checked = 0;
    for (const auto& state : states()) {
        const auto bytes = packed(state);
        const auto copy = unpack(bytes);
        ASSERT_EQ(packed(copy), bytes);
        expect_plays_alike(state, copy, checked++);
    }
    EXPECT_GT(checked, 1000);
}

TEST(StateCodec, RejectsTruncatedAndTrailingBytes) {
    const auto bytes = packed(states().front());
    for (std::size_t n = 0; n < bytes.size(); ++n)
        EXPECT_THROW(unpack({bytes.data(), n}), std::invalid_argument) << n << " of " << bytes.size() << " bytes";
    EXPECT_THROW(unpack(bytes + '\0'), std::invalid_argument);
}

struct StoreFile {
    std::filesystem::path path;

    StoreFile() {
        auto name = (std::filesystem::temp_directory_path() / "combat-state-XXXXXX").string();
        const int fd = mkstemp(name.data());
        if (fd < 0) throw std::runtime_error{"cannot create test store"};
        close(fd);
        path = name;
    }
    ~StoreFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

TEST(StateStore, ReadsEveryRecordInAnyOrderFromSeveralThreads) {
    const StoreFile file;
    std::vector<const sts::BattleContext*> all;
    {
        StateWriter writer{file.path};
        for (const auto& state : states()) {
            EXPECT_EQ(writer.append(state), all.size());
            all.push_back(&state);
        }
        writer.flush();
    }
    const StateReader reader{file.path};
    ASSERT_EQ(reader.size(), all.size());
    std::vector<std::size_t> order(all.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::shuffle(order, std::mt19937_64{7});
    std::vector<std::future<void>> readers;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        readers.push_back(std::async(std::launch::async, [&, lane] {
            for (std::size_t j = lane; j < order.size(); j += 4) {
                const auto i = order[j];
                ASSERT_EQ(packed(reader.at(i)), packed(*all[i])) << "record " << i;
            }
        }));
    }
    for (auto& task : readers) task.get();
}

TEST(StateStore, FlushMakesAppendedRecordsReadable) {
    const StoreFile file;
    StateWriter writer{file.path};
    writer.flush();
    EXPECT_EQ(StateReader{file.path}.size(), 0);
    EXPECT_EQ(writer.append(states().front()), 0);
    writer.flush();
    const StateReader reader{file.path};
    EXPECT_EQ(reader.size(), writer.size());
    EXPECT_EQ(packed(reader.at(0)), packed(states().front()));
    EXPECT_THROW(reader.at(reader.size()), std::out_of_range);
}

TEST(StateStore, RestoredNestedChoicesResumeIdentically) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, 17, 20};
    sts::BattleContext state;
    state.init(game, sts::MonsterEncounter::JAW_WORM);
    state.cards = sts::CardManager{};
    state.cards.createTempCardInHand(sts::CardInstance{sts::CardId::HAVOC});
    state.cards.createTempCardInDrawPile(0, sts::CardInstance{sts::CardId::HEADBUTT});
    state.cards.createTempCardInDiscard(sts::CardInstance{sts::CardId::DEFEND_RED});
    state.cards.createTempCardInDiscard(sts::CardInstance{sts::CardId::BASH});
    sts::search::Action{sts::search::ActionType::CARD, 0, 0}.execute(state);
    ASSERT_EQ(state.inputState, sts::InputState::CARD_SELECT);
    ASSERT_GT(state.resolution.size(), 1);

    const StoreFile file;
    StateWriter writer{file.path};
    writer.append(state);
    writer.flush();
    const StateReader reader{file.path};
    const auto restored = reader.at(0);
    ASSERT_EQ(packed(restored), packed(state));
    expect_plays_alike(state, restored, 1);
}

TEST(StateStore, TruncatedFileIsRejected) {
    const StoreFile file;
    {
        StateWriter writer{file.path};
        writer.append(states().front());
    }
    std::filesystem::resize_file(file.path, std::filesystem::file_size(file.path) - 1);
    EXPECT_THROW(StateReader{file.path}, std::runtime_error);
}
}  // namespace
}  // namespace stsrl::combat
