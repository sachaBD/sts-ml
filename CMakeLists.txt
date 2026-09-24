cmake_minimum_required(VERSION 3.25)

project(sts_combat_rl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared simulator repo (also used by sts_ml).
set(STS_LIGHTSPEED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sts_lightspeed" CACHE PATH "sts_lightspeed checkout")
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sts_lightspeed")

add_library(combat_core STATIC
    agents/mcts_agent.cpp
    agents/random_agent.cpp
    agents/teacher_leaves.cpp
    agents/teacher_search.cpp
    models/value_net.cpp
    combat/environment.cpp
    scenarios/act1_run.cpp
    scenarios/jaw_worm.cpp
    scenarios/slime_boss.cpp
    scenarios/slime_entry_projection.cpp)
target_include_directories(combat_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(combat_core PUBLIC sts_lightspeed)
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

add_executable(play_entry_mcts_rollout apps/play_entry_mcts_rollout.cpp)
target_link_libraries(play_entry_mcts_rollout PRIVATE combat_core)
target_compile_features(play_entry_mcts_rollout PRIVATE cxx_std_23)
target_compile_options(play_entry_mcts_rollout PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_entry_mcts_neural apps/play_entry_mcts_neural.cpp)
target_link_libraries(play_entry_mcts_neural PRIVATE combat_core)
target_compile_features(play_entry_mcts_neural PRIVATE cxx_std_23)
target_compile_options(play_entry_mcts_neural PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_slime_mcts_neural apps/play_slime_mcts_neural.cpp)
target_link_libraries(play_slime_mcts_neural PRIVATE combat_core)
target_compile_features(play_slime_mcts_neural PRIVATE cxx_std_23)
target_compile_options(play_slime_mcts_neural PRIVATE -Wall -Wextra -Wpedantic)

add_executable(smoke_slime_entry_projections apps/smoke_slime_entry_projections.cpp)
target_link_libraries(smoke_slime_entry_projections PRIVATE combat_core)
target_compile_features(smoke_slime_entry_projections PRIVATE cxx_std_23)
target_compile_options(smoke_slime_entry_projections PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bootstrap_fight_worker apps/bootstrap/fight_worker.cpp)
target_link_libraries(bootstrap_fight_worker PRIVATE combat_core)
target_compile_features(bootstrap_fight_worker PRIVATE cxx_std_23)
target_compile_options(bootstrap_fight_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(value_play_worker apps/value_play/worker.cpp)
target_link_libraries(value_play_worker PRIVATE combat_core)
target_compile_features(value_play_worker PRIVATE cxx_std_23)
target_compile_options(value_play_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dagger_worker apps/dagger/worker.cpp)
target_link_libraries(dagger_worker PRIVATE combat_core)
target_compile_features(dagger_worker PRIVATE cxx_std_23)
target_compile_options(dagger_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(play_entry_pbcs apps/play_entry_pbcs.cpp)
target_link_libraries(play_entry_pbcs PRIVATE combat_core)
target_compile_features(play_entry_pbcs PRIVATE cxx_std_23)
target_compile_options(play_entry_pbcs PRIVATE -Wall -Wextra -Wpedantic)

add_executable(value_net_eval apps/value_net_eval.cpp)
target_link_libraries(value_net_eval PRIVATE combat_core)
target_compile_features(value_net_eval PRIVATE cxx_std_23)
target_compile_options(value_net_eval PRIVATE -Wall -Wextra -Wpedantic)

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

add_executable(teacher_leaves_test tests/teacher_leaves_test.cpp)
target_link_libraries(teacher_leaves_test PRIVATE combat_core)
target_compile_features(teacher_leaves_test PRIVATE cxx_std_23)
target_compile_options(teacher_leaves_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME teacher_leaves_test COMMAND teacher_leaves_test)

# The native value net's inner loops benefit from AVX2/FMA on the machine that runs it.
set_source_files_properties(models/value_net.cpp PROPERTIES COMPILE_OPTIONS "-march=native")
