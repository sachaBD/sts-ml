#include "agents/combat/player/play_config.hpp"
#include <gtest/gtest.h>
#include <string>

using stsrl::player::parse_play_config;

namespace {
const std::string valid = R"(
[search]
sims = 64
particles = 8
gumbel_scale = 1
rollout_mix = 0.5

[workers]
actors = 2
slots = 3
gpu_wait_us = 100
timeout_ms = 60000
shutdown_timeout_ms = 5000
)";

std::string with(const std::string& from, const std::string& to) {
    auto text = valid;
    text.replace(text.find(from), from.size(), to);
    return text;
}

void rejects(const std::string& text, const std::string& message) {
    try {
        parse_play_config(text);
        FAIL() << "accepted: " << message;
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string{e.what()}.find(message), std::string::npos) << e.what();
    }
}
}  // namespace

TEST(PlayConfig, EveryKeyReachesItsSetting) {
    const auto c = parse_play_config(valid);
    EXPECT_EQ(c.search.sims, 64);
    EXPECT_EQ(c.search.particles, 8);
    EXPECT_EQ(c.search.gumbel_scale, 1.0);
    EXPECT_EQ(c.search.rollout_mix, 0.5);
    EXPECT_EQ(c.workers.actors, 2);
    EXPECT_EQ(c.workers.slots, 3);
    EXPECT_EQ(c.workers.batch_wait.count(), 100);
    EXPECT_EQ(c.workers.timeout.count(), 60000);
    EXPECT_EQ(c.workers.shutdown_timeout.count(), 5000);
}

TEST(PlayConfig, AMistakeNamesItsSetting) {
    rejects(with("sims = 64\n", ""), "missing [search].sims");
    rejects(with("sims = 64", "sims = 64\nsimz = 1"), "unknown [search].simz");
    rejects(with("actors = 2", "actors = 2.5"), "[workers].actors has the wrong type");
    rejects(with("actors = 2", "actors = \"2\""), "[workers].actors has the wrong type");
    rejects(with("[workers]", "[workerz]"), "missing [workers]");
    rejects(valid + "[extra]\n", "unknown [extra]");
    rejects(with("rollout_mix = 0.5", "rollout_mix = 2"), "search:");
    rejects(with("slots = 3", "slots = 0"), "workers:");
}
