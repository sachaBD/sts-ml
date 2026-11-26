# Combat searches know the simulator and nothing else: no network, inference, threads or files.
add_library(combat_search STATIC
    gumbel/gumbel.cpp
    paired_turn/paired_turn.cpp
    rollout_blend/rollout_blend.cpp
    config.cpp)
target_link_libraries(combat_search PUBLIC combat_environment)
target_compile_options(combat_search PRIVATE -Wall -Wextra -Wpedantic)

add_executable(gumbel_test gumbel/gumbel_test.cpp)
target_link_libraries(gumbel_test PRIVATE combat_search combat_scenarios GTest::gtest_main)
target_compile_options(gumbel_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(gumbel_test PROPERTIES TIMEOUT 60)

add_executable(paired_turn_test paired_turn/paired_turn_test.cpp)
target_link_libraries(paired_turn_test PRIVATE combat_search combat_scenarios GTest::gtest_main)
target_compile_options(paired_turn_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(paired_turn_test PROPERTIES TIMEOUT 10)
