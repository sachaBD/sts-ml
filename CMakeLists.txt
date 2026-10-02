cmake_minimum_required(VERSION 3.25)

project(sts_combat_rl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link-time optimization for everything below, sts_lightspeed included: its many small out-of-line accessors
# (CardInstance / Monster getters) get inlined across files. ~1.47x search throughput, identical results
# (scratch/profile_teacher, 2026-09-29). Opt out with -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "LTO not supported, building without it: ${ipo_error}")
    endif()
endif()

# Shared simulator repo (also used by sts_ml).
set(STS_LIGHTSPEED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sts_lightspeed" CACHE PATH "sts_lightspeed checkout")
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sts_lightspeed")

add_library(combat_core STATIC
    agents/combat/search/teacher_leaves.cpp
    agents/combat/search/teacher_search.cpp
    agents/combat/value/value_net.cpp
    environments/combat/environment.cpp
    environments/combat/encoding_v4.cpp
    environments/combat/battle_snapshot.cpp
    environments/overworld/act1_run.cpp
    environments/combat/scenarios/jaw_worm.cpp
    environments/combat/scenarios/slime_boss.cpp)
target_include_directories(combat_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(combat_core PUBLIC sts_lightspeed)
target_compile_features(combat_core PUBLIC cxx_std_23)
target_compile_options(combat_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bootstrap_fight_worker apps/bootstrap/fight_worker.cpp)
target_link_libraries(bootstrap_fight_worker PRIVATE combat_core)
target_compile_features(bootstrap_fight_worker PRIVATE cxx_std_23)
target_compile_options(bootstrap_fight_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(value_play_worker apps/value_play/worker.cpp)
target_link_libraries(value_play_worker PRIVATE combat_core)
target_compile_features(value_play_worker PRIVATE cxx_std_23)
target_compile_options(value_play_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fight_resample_worker apps/fight_resample/worker.cpp)
target_link_libraries(fight_resample_worker PRIVATE combat_core)
target_compile_features(fight_resample_worker PRIVATE cxx_std_23)
target_compile_options(fight_resample_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(export_save apps/export_save/export_save.cpp)
target_link_libraries(export_save PRIVATE combat_core)
target_compile_features(export_save PRIVATE cxx_std_23)
target_compile_options(export_save PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dagger_worker apps/dagger/worker.cpp)
target_link_libraries(dagger_worker PRIVATE combat_core)
target_compile_features(dagger_worker PRIVATE cxx_std_23)
target_compile_options(dagger_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(gauntlet_worker apps/gauntlet/worker.cpp)
target_link_libraries(gauntlet_worker PRIVATE combat_core)
target_compile_features(gauntlet_worker PRIVATE cxx_std_23)
target_compile_options(gauntlet_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(card_marginals_worker apps/card_marginals/worker.cpp)
target_link_libraries(card_marginals_worker PRIVATE combat_core)
target_compile_features(card_marginals_worker PRIVATE cxx_std_23)
target_compile_options(card_marginals_worker PRIVATE -Wall -Wextra -Wpedantic)

# Research benchmark retained as an opt-in target, not a runnable app.
add_executable(bench_reuse EXCLUDE_FROM_ALL experiments/reuse_probe/bench_reuse.cpp)
target_link_libraries(bench_reuse PRIVATE combat_core)

enable_testing()
add_executable(encoding_test environments/combat/encoding_test.cpp)
target_link_libraries(encoding_test PRIVATE combat_core)
target_compile_features(encoding_test PRIVATE cxx_std_23)
target_compile_options(encoding_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME encoding_test COMMAND encoding_test)

add_executable(encoding_v4_test environments/combat/encoding_v4_test.cpp)
target_link_libraries(encoding_v4_test PRIVATE combat_core)
target_compile_features(encoding_v4_test PRIVATE cxx_std_23)
target_compile_options(encoding_v4_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME encoding_v4_test COMMAND encoding_v4_test)

add_executable(slime_boss_test environments/combat/scenarios/slime_boss_test.cpp)
target_link_libraries(slime_boss_test PRIVATE combat_core)
target_compile_features(slime_boss_test PRIVATE cxx_std_23)
target_compile_options(slime_boss_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME slime_boss_test COMMAND slime_boss_test)

add_executable(teacher_leaves_test agents/combat/search/teacher_leaves_test.cpp)
target_link_libraries(teacher_leaves_test PRIVATE combat_core)
target_compile_features(teacher_leaves_test PRIVATE cxx_std_23)
target_compile_options(teacher_leaves_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME teacher_leaves_test COMMAND teacher_leaves_test)

add_executable(search_perf_test agents/combat/search/search_perf_test.cpp)
target_link_libraries(search_perf_test PRIVATE combat_core)
target_compile_features(search_perf_test PRIVATE cxx_std_23)
target_compile_options(search_perf_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME search_perf_test COMMAND search_perf_test)

add_executable(search_priors_test agents/combat/search/search_priors_test.cpp)
target_link_libraries(search_priors_test PRIVATE combat_core)
target_compile_features(search_priors_test PRIVATE cxx_std_23)
target_compile_options(search_priors_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME search_priors_test COMMAND search_priors_test)

# The native value net's inner loops benefit from AVX2/FMA on the machine that runs it.
set_source_files_properties(agents/combat/value/value_net.cpp PROPERTIES COMPILE_OPTIONS "-march=native")

add_executable(combat_transition_worker apps/combat_transition/worker.cpp)
target_link_libraries(combat_transition_worker PRIVATE combat_core)
target_compile_features(combat_transition_worker PRIVATE cxx_std_23)
target_compile_options(combat_transition_worker PRIVATE -Wall -Wextra -Wpedantic)

# Macro simulator (out-of-combat act play with injected fight results): environments/overworld/macro_sim.hpp.
add_library(macro_sim STATIC
    environments/overworld/macro_sim.cpp
    environments/overworld/decisions.cpp)
target_include_directories(macro_sim PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(macro_sim PUBLIC sts_lightspeed)
target_compile_features(macro_sim PUBLIC cxx_std_23)
target_compile_options(macro_sim PRIVATE -Wall -Wextra -Wpedantic)
add_executable(macro_sim_check environments/overworld/check.cpp)
target_link_libraries(macro_sim_check PRIVATE macro_sim)
target_compile_options(macro_sim_check PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME macro_sim_paths_test COMMAND macro_sim_check paths 3)

# Card-pick search worker (docs/research/card-policy-loop): rollouts per card option, fights/picks batched to Python.
add_library(overworld_search STATIC agents/overworld/search/rollouts.cpp)
target_link_libraries(overworld_search PUBLIC macro_sim)
target_compile_options(overworld_search PRIVATE -Wall -Wextra -Wpedantic)

add_executable(card_search_worker apps/card_search/worker.cpp)
target_link_libraries(card_search_worker PRIVATE overworld_search)
target_compile_options(card_search_worker PRIVATE -Wall -Wextra -Wpedantic)

# Real-run RL worker (docs/research/run-rl): real act 1 runs, MCTS fights, card picks asked of Python.
add_executable(run_rl_worker apps/run_rl/worker.cpp)
target_link_libraries(run_rl_worker PRIVATE combat_core macro_sim)
target_compile_features(run_rl_worker PRIVATE cxx_std_23)
target_compile_options(run_rl_worker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(run_rl_worker_test apps/run_rl/worker_test.cpp)
target_link_libraries(run_rl_worker_test PRIVATE combat_core macro_sim)
target_compile_features(run_rl_worker_test PRIVATE cxx_std_23)
target_compile_options(run_rl_worker_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME run_rl_worker_test COMMAND run_rl_worker_test)

# Colocated Python tests share CTest with native checks. Never recurse through
# runs/schema=*/ output trees: only runs/test_*.py is source-level test code.
set(STSRL_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3" CACHE FILEPATH "Python interpreter for tests")
file(GLOB_RECURSE python_test_files CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "agents/test_*.py" "environments/test_*.py" "models/test_*.py" "apps/test_*.py")
file(GLOB run_test_files CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "runs/test_*.py")
list(APPEND python_test_files ${run_test_files})
file(RELATIVE_PATH python_test_build_dir "${CMAKE_CURRENT_SOURCE_DIR}/build" "${CMAKE_CURRENT_BINARY_DIR}")
foreach(test_file IN LISTS python_test_files)
    string(REGEX REPLACE "\\.py$" "" test_module "${test_file}")
    string(REPLACE "/" "." test_module "${test_module}")
    add_test(NAME "python.${test_module}" COMMAND "${STSRL_PYTHON}" -m "${test_module}")
    set_tests_properties("python.${test_module}" PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};STSRL_BUILD_DIR=${python_test_build_dir}"
        LABELS "python")
endforeach()

add_executable(battle_snapshot_test environments/combat/battle_snapshot_test.cpp)
target_link_libraries(battle_snapshot_test PRIVATE combat_core)
target_compile_features(battle_snapshot_test PRIVATE cxx_std_23)
add_test(NAME battle_snapshot_test COMMAND battle_snapshot_test)

# Public/history-reconstructible input contract for the new overworld graph topology.
add_executable(overworld_observation_test environments/overworld/observation_test.cpp)
target_link_libraries(overworld_observation_test PRIVATE macro_sim)
target_compile_options(overworld_observation_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME overworld_observation_test COMMAND overworld_observation_test)
