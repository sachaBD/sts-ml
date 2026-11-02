#include "agents/combat/nn/runtime_telemetry.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

TEST(RuntimeTelemetry, AtomicSnapshotsPreserveIdentityAndReplaceRatherThanAppend) {
    const auto directory = std::filesystem::temp_directory_path() /
        ("stsrl-telemetry-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    const auto path = directory / "runtime.json";
    auto observer = stsrl::nn::runtime_observer(path.c_str());
    stsrl::infra::RuntimeSnapshot snapshot;
    snapshot.elapsed_seconds = 1;
    snapshot.captured_unix_seconds = 100;
    observer(snapshot);
    nlohmann::json first; { std::ifstream stream{path}; stream >> first; }
    snapshot.sequence = 1; snapshot.elapsed_seconds = 2;
    observer(snapshot);
    nlohmann::json second; { std::ifstream stream{path}; stream >> second; }
    EXPECT_EQ(first["worker_instance_id"], second["worker_instance_id"]);
    EXPECT_EQ(second["sequence"], 1);
    EXPECT_EQ(second["schema_version"], 1);
    EXPECT_TRUE(second["actors"].is_array());
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp." + std::to_string(::getpid())));
    EXPECT_FALSE(stsrl::nn::runtime_observer(nullptr));
    EXPECT_FALSE(stsrl::nn::runtime_observer(""));
}
