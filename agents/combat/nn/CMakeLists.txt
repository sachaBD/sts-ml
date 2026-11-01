# Combat owns episodes/search/encoding; infra owns workers and inference.
add_library(nn_decision STATIC decision.cpp explanation.cpp tokens.cpp)
target_link_libraries(nn_decision PUBLIC combat_core combat_search)
target_compile_options(nn_decision PRIVATE -Wall -Wextra -Wpedantic)

foreach(test decision_test explanation_test)
    add_executable(nn_${test} ${test}.cpp)
    target_link_libraries(nn_${test} PRIVATE nn_decision GTest::gtest_main)
    target_compile_options(nn_${test} PRIVATE -Wall -Wextra -Wpedantic)
    gtest_discover_tests(nn_${test} PROPERTIES TIMEOUT 10)
endforeach()

if(NOT TARGET infra_inference)
    message(STATUS "nn_play / nn_explain not built: infra_inference unavailable")
    return()
endif()
add_executable(nn_play play.cpp)
target_link_libraries(nn_play PRIVATE nn_decision infra_actors infra_inference)
target_compile_options(nn_play PRIVATE -Wall -Wextra -Wpedantic)

add_executable(nn_explain explain.cpp)
target_link_libraries(nn_explain PRIVATE nn_decision infra_inference)
target_compile_options(nn_explain PRIVATE -Wall -Wextra -Wpedantic)
