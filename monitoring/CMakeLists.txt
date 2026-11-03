# The telemetry format's writer (header-only): run_actors Counters as JSON lines.
add_library(monitoring INTERFACE)
target_include_directories(monitoring SYSTEM INTERFACE "${STS_LIGHTSPEED_DIR}/json/include")
target_link_libraries(monitoring INTERFACE infra_actors)
