add_library(combat_inspection STATIC inspection.cpp)
target_link_libraries(combat_inspection PUBLIC player_decision)
target_compile_options(combat_inspection PRIVATE -Wall -Wextra -Wpedantic)

add_executable(combat_inspection_test inspection_test.cpp)
target_link_libraries(combat_inspection_test PRIVATE combat_inspection GTest::gtest_main)
target_compile_options(combat_inspection_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(combat_inspection_test PROPERTIES TIMEOUT 30)

if(TARGET infra_inference)
    add_executable(combat_explain explain.cpp)
    target_link_libraries(combat_explain PRIVATE combat_inspection nn_network infra_actors CLI11::CLI11)
    target_compile_options(combat_explain PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(combat_lab lab.cpp)
    target_link_libraries(combat_lab PRIVATE combat_inspection nn_network CLI11::CLI11)
    target_compile_options(combat_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()
