# The agent: a search guided by the network, playing fights (nn_play).
add_library(player_decision STATIC decision.cpp targets.cpp)
target_link_libraries(player_decision PUBLIC nn_tokens combat_search)
target_compile_options(player_decision PRIVATE -Wall -Wextra -Wpedantic)

# Fights as infra episodes, recording each finished one.
add_library(player_episodes STATIC episodes.cpp)
target_link_libraries(player_episodes PUBLIC player_decision infra_actors)
target_compile_options(player_episodes PRIVATE -Wall -Wextra -Wpedantic)

# toml++ reads nn_play's settings (play_config), pinned and built with the project.
FetchContent_Declare(tomlplusplus URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(tomlplusplus)

# nn_play's settings file.
add_library(player_config STATIC play_config.cpp)
target_link_libraries(player_config PUBLIC combat_search infra_actors PRIVATE tomlplusplus::tomlplusplus)
target_compile_options(player_config PRIVATE -Wall -Wextra -Wpedantic)

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
# CLI11 parses nn_play's command line, pinned and built with the project.
FetchContent_Declare(cli11 URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.5.0.tar.gz
    URL_HASH SHA256=17e02b4cddc2fa348e5dbdbb582c59a3486fa2b2433e70a0c3bacb871334fd55 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(cli11)

add_executable(nn_play nn_play.cpp)
target_link_libraries(nn_play PRIVATE player_episodes player_config nn_network CLI11::CLI11)
target_compile_options(nn_play PRIVATE -Wall -Wextra -Wpedantic)
