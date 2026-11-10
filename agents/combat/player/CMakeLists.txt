# The agent: a search guided by the network, playing fights (nn_play).
add_library(player_decision STATIC decision.cpp targets.cpp)
target_link_libraries(player_decision PUBLIC nn_tokens combat_search)
target_compile_options(player_decision PRIVATE -Wall -Wextra -Wpedantic)

foreach(test decision_test targets_test)
    add_executable(player_${test} ${test}.cpp)
    target_link_libraries(player_${test} PRIVATE player_decision combat_scenarios GTest::gtest_main)
    target_compile_options(player_${test} PRIVATE -Wall -Wextra -Wpedantic)
    gtest_discover_tests(player_${test} PROPERTIES TIMEOUT 10)
endforeach()

if(NOT TARGET infra_inference)
    message(STATUS "nn_play not built: infra_inference unavailable")
    return()
endif()
add_executable(nn_play play.cpp)
target_link_libraries(nn_play PRIVATE player_decision infra_actors infra_inference)
target_compile_options(nn_play PRIVATE -Wall -Wextra -Wpedantic)
