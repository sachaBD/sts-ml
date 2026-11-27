# Combat searches know the simulator and nothing else: no network, inference, threads or files.
add_library(combat_search STATIC
    gumbel/gumbel.cpp
    sift_sieve/sift_sieve.cpp
    rollout_blend/rollout_blend.cpp
    config.cpp)
target_link_libraries(combat_search PUBLIC combat_environment)
target_compile_options(combat_search PRIVATE -Wall -Wextra -Wpedantic)

add_executable(gumbel_test gumbel/gumbel_test.cpp)
target_link_libraries(gumbel_test PRIVATE combat_search combat_scenarios GTest::gtest_main)
target_compile_options(gumbel_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(gumbel_test PROPERTIES TIMEOUT 60)

add_executable(sift_sieve_test sift_sieve/sift_sieve_test.cpp)
target_link_libraries(sift_sieve_test PRIVATE combat_search combat_scenarios GTest::gtest_main)
target_compile_options(sift_sieve_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(sift_sieve_test PROPERTIES TIMEOUT 10)
