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

# Python training consumes the same native encoders as inference.
set_target_properties(combat_environment combat_search nn_tokens PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(Python3_EXECUTABLE "${STSRL_PYTHON}")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${STSRL_PYTHON}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_encoding_bridge encoding_bridge.cpp)
target_link_libraries(_encoding_bridge PRIVATE nn_tokens)
target_compile_options(_encoding_bridge PRIVATE -Wall -Wextra -Wpedantic)

add_executable(encoding_bridge_fixture encoding_bridge_fixture.cpp)
target_link_libraries(encoding_bridge_fixture PRIVATE nn_tokens combat_scenarios)
target_compile_options(encoding_bridge_fixture PRIVATE -Wall -Wextra -Wpedantic)

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
    target_link_libraries(nn_network PUBLIC nn_tokens infra_inference PRIVATE tomlplusplus::tomlplusplus)
    target_compile_options(nn_network PRIVATE -Wall -Wextra -Wpedantic)
endif()
