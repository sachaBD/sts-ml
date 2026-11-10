# layout.hpp: the token layouts as C++ constants, generated from layout.toml.
set(NN_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(OUTPUT "${NN_GENERATED}/agents/combat/nn/layout.hpp"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NN_GENERATED}/agents/combat/nn"
    COMMAND "${STSRL_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/layout.py" "${NN_GENERATED}/agents/combat/nn/layout.hpp"
    DEPENDS layout.toml layout.py
    COMMENT "Generating agents/combat/nn/layout.hpp")

# Network inputs: the shared token format plus each model's encoding.
add_library(nn_tokens STATIC features.cpp tokens.cpp entity/encoding.cpp inventory/encoding.cpp
    "${NN_GENERATED}/agents/combat/nn/layout.hpp")
target_include_directories(nn_tokens PUBLIC "${NN_GENERATED}")
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

# An exported model directory as inference backends.
if(TARGET infra_inference)
    add_library(nn_network STATIC network.cpp)
    target_link_libraries(nn_network PUBLIC nn_tokens infra_inference)
    target_include_directories(nn_network SYSTEM PRIVATE "${STS_LIGHTSPEED_DIR}/json/include")
    target_compile_options(nn_network PRIVATE -Wall -Wextra -Wpedantic)
endif()
