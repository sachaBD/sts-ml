# toml++ reads each encoding's layout.toml, pinned and built with the project.
FetchContent_Declare(tomlplusplus URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(tomlplusplus)

# Network inputs: each encoding, written as its layout.toml describes.
add_library(nn_tokens STATIC features.cpp layout.cpp tokens.cpp entity/encoding.cpp inventory/encoding.cpp)
target_link_libraries(nn_tokens PUBLIC combat_environment combat_search PRIVATE tomlplusplus::tomlplusplus)
target_compile_options(nn_tokens PRIVATE -Wall -Wextra -Wpedantic)

add_executable(nn_features_test features_test.cpp)
target_link_libraries(nn_features_test PRIVATE nn_tokens combat_scenarios GTest::gtest_main)
target_compile_options(nn_features_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(nn_features_test PROPERTIES TIMEOUT 10)

add_executable(nn_inventory_encoding_test inventory/encoding_test.cpp)
target_link_libraries(nn_inventory_encoding_test PRIVATE nn_tokens combat_scenarios GTest::gtest_main)
target_compile_definitions(nn_inventory_encoding_test PRIVATE STSRL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_options(nn_inventory_encoding_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(nn_inventory_encoding_test PROPERTIES TIMEOUT 10)

# An exported model directory as inference backends.
if(TARGET infra_inference)
    add_library(nn_network STATIC network.cpp)
    target_link_libraries(nn_network PUBLIC nn_tokens infra_inference)
    target_include_directories(nn_network SYSTEM PRIVATE "${STS_LIGHTSPEED_DIR}/json/include")
    target_compile_options(nn_network PRIVATE -Wall -Wextra -Wpedantic)
endif()
