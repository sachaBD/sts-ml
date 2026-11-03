# The telemetry format's writer (header-only): run_actors Counters as JSON lines.
add_library(monitoring INTERFACE)
target_include_directories(monitoring SYSTEM INTERFACE "${STS_LIGHTSPEED_DIR}/json/include")
target_link_libraries(monitoring INTERFACE infra_actors)

add_executable(telemetry_test telemetry_test.cpp)
target_link_libraries(telemetry_test PRIVATE monitoring GTest::gtest_main)
target_compile_options(telemetry_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(telemetry_test PROPERTIES TIMEOUT 10)
