cmake_minimum_required(VERSION 3.25)

project(sts_combat_rl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE STS_LIGHTSPEED_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/sts_lightspeed/src/*.cpp")

add_library(sts_lightspeed_core STATIC ${STS_LIGHTSPEED_SOURCES})
target_include_directories(sts_lightspeed_core
    SYSTEM PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/sts_lightspeed/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/sts_lightspeed/json/include")
target_compile_options(sts_lightspeed_core PRIVATE -Wno-shift-count-overflow)

add_library(combat_core STATIC
    agents/mcts_agent.cpp
    agents/random_agent.cpp
    combat/environment.cpp
    scenarios/jaw_worm.cpp
    scenarios/slime_boss.cpp
    scenarios/slime_entry_projection.cpp)
target_include_directories(combat_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(combat_core PUBLIC sts_lightspeed_core)
target_compile_features(combat_core PUBLIC cxx_std_23)
target_compile_options(combat_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_jaw_worm apps/play_jaw_worm.cpp)
target_link_libraries(play_jaw_worm PRIVATE combat_core)
target_compile_features(play_jaw_worm PRIVATE cxx_std_23)
target_compile_options(play_jaw_worm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_jaw_worm_mcts apps/play_jaw_worm_mcts.cpp)
target_link_libraries(play_jaw_worm_mcts PRIVATE combat_core)
target_compile_features(play_jaw_worm_mcts PRIVATE cxx_std_23)
target_compile_options(play_jaw_worm_mcts PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_slime_mcts_rollout apps/play_slime_mcts_rollout.cpp)
target_link_libraries(play_slime_mcts_rollout PRIVATE combat_core)
target_compile_features(play_slime_mcts_rollout PRIVATE cxx_std_23)
target_compile_options(play_slime_mcts_rollout PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_slime_mcts_neural apps/play_slime_mcts_neural.cpp)
target_link_libraries(play_slime_mcts_neural PRIVATE combat_core)
target_compile_features(play_slime_mcts_neural PRIVATE cxx_std_23)
target_compile_options(play_slime_mcts_neural PRIVATE -Wall -Wextra -Wpedantic)

add_executable(smoke_slime_entry_projections apps/smoke_slime_entry_projections.cpp)
target_link_libraries(smoke_slime_entry_projections PRIVATE combat_core)
target_compile_features(smoke_slime_entry_projections PRIVATE cxx_std_23)
target_compile_options(smoke_slime_entry_projections PRIVATE -Wall -Wextra -Wpedantic)

add_executable(generate_entry_mcts_records apps/generate_entry_mcts_records.cpp)
target_link_libraries(generate_entry_mcts_records PRIVATE combat_core)
target_compile_features(generate_entry_mcts_records PRIVATE cxx_std_23)
target_compile_options(generate_entry_mcts_records PRIVATE -Wall -Wextra -Wpedantic)

add_executable(generate_mcts_records apps/generate_mcts_records.cpp)
target_link_libraries(generate_mcts_records PRIVATE combat_core)
target_compile_features(generate_mcts_records PRIVATE cxx_std_23)
target_compile_options(generate_mcts_records PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(encoding_test tests/encoding_test.cpp)
target_link_libraries(encoding_test PRIVATE combat_core)
target_compile_features(encoding_test PRIVATE cxx_std_23)
target_compile_options(encoding_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME encoding_test COMMAND encoding_test)

add_executable(slime_boss_test tests/slime_boss_test.cpp)
target_link_libraries(slime_boss_test PRIVATE combat_core)
target_compile_features(slime_boss_test PRIVATE cxx_std_23)
target_compile_options(slime_boss_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME slime_boss_test COMMAND slime_boss_test)

add_executable(mcts_leaf_evaluator_test tests/mcts_leaf_evaluator_test.cpp)
target_link_libraries(mcts_leaf_evaluator_test PRIVATE combat_core)
target_compile_features(mcts_leaf_evaluator_test PRIVATE cxx_std_23)
target_compile_options(mcts_leaf_evaluator_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME mcts_leaf_evaluator_test COMMAND mcts_leaf_evaluator_test)

add_executable(slime_entry_projection_test tests/slime_entry_projection_test.cpp)
target_link_libraries(slime_entry_projection_test PRIVATE combat_core)
target_compile_features(slime_entry_projection_test PRIVATE cxx_std_23)
target_compile_options(slime_entry_projection_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME slime_entry_projection_test COMMAND slime_entry_projection_test)
