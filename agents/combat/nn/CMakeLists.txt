# Combat owns episodes/search/encoding; infra owns workers and inference.
add_library(nn_decision STATIC decision.cpp tokens.cpp)
target_link_libraries(nn_decision PUBLIC combat_core combat_search)
target_compile_options(nn_decision PRIVATE -Wall -Wextra -Wpedantic)

add_executable(nn_decision_test decision_test.cpp)
target_link_libraries(nn_decision_test PRIVATE nn_decision GTest::gtest_main)
target_compile_options(nn_decision_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(nn_decision_test PROPERTIES TIMEOUT 10)

if(NOT TARGET infra_inference)
    message(STATUS "nn_play not built: infra_inference unavailable")
    return()
endif()
add_executable(nn_play play.cpp)
target_link_libraries(nn_play PRIVATE nn_decision infra_actors infra_inference)
target_compile_options(nn_play PRIVATE -Wall -Wextra -Wpedantic)
