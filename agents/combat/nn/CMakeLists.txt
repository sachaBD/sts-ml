# Network inputs: the shared token format plus each model's encoding.
add_library(nn_tokens STATIC features.cpp tokens.cpp entity/encoding.cpp inventory/encoding.cpp)
target_link_libraries(nn_tokens PUBLIC combat_environment combat_search)
target_compile_options(nn_tokens PRIVATE -Wall -Wextra -Wpedantic)

add_executable(nn_features_test features_test.cpp)
target_link_libraries(nn_features_test PRIVATE nn_tokens combat_scenarios GTest::gtest_main)
target_compile_options(nn_features_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(nn_features_test PROPERTIES TIMEOUT 10)

add_executable(nn_inventory_encoding_test inventory/encoding_test.cpp)
target_link_libraries(nn_inventory_encoding_test PRIVATE nn_tokens combat_scenarios GTest::gtest_main)
target_compile_options(nn_inventory_encoding_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(nn_inventory_encoding_test PROPERTIES TIMEOUT 10)
