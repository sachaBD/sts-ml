#include "monitoring/telemetry.hpp"
#include <gtest/gtest.h>

using namespace stsrl;

TEST(TelemetryFile, QueueGaugesSurviveSerializationAndDrain) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("queue-telemetry-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
    infra::Counters counters;
    counters.time = std::chrono::system_clock::now();
    counters.queues = {{7, std::chrono::duration<double>{.25}, 3, 5}, {0, {}, 0, 0}};
    {
        monitoring::TelemetryFile file{path};
        file.write(counters);
        counters.queues[0].depth = 0;
        file.write(counters);
    }
    std::ifstream input{path};
    for (const auto depth : {3, 0}) {
        std::string line;
        ASSERT_TRUE(std::getline(input, line));
        const auto queues = nlohmann::json::parse(line).at("queues");
        EXPECT_EQ(queues.at(0).at("depth"), depth);
        EXPECT_EQ(queues.at(0).at("peak_depth"), 5);
        EXPECT_EQ(queues.at(0).at("requests"), 7);
        EXPECT_EQ(queues.at(1).at("depth"), 0);
        EXPECT_EQ(queues.at(1).at("peak_depth"), 0);
    }
    std::filesystem::remove(path);
}
