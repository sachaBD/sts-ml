cmake_minimum_required(VERSION 3.25)

project(sts_combat_rl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared simulator repo (also used by sts_ml).
set(STS_LIGHTSPEED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sts_lightspeed" CACHE PATH "sts_lightspeed checkout")
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sts_lightspeed")

add_library(combat_core STATIC
    agents/teacher_leaves.cpp
    agents/teacher_search.cpp
    topology/value_net.cpp
    combat/environment.cpp
    scenarios/act1_run.cpp
    scenarios/jaw_worm.cpp
    scenarios/slime_boss.cpp)
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

add_executable(bench_reuse apps/bench_reuse.cpp)
target_link_libraries(bench_reuse PRIVATE combat_core)

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

add_executable(teacher_leaves_test tests/teacher_leaves_test.cpp)
target_link_libraries(teacher_leaves_test PRIVATE combat_core)
target_compile_features(teacher_leaves_test PRIVATE cxx_std_23)
target_compile_options(teacher_leaves_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME teacher_leaves_test COMMAND teacher_leaves_test)

add_executable(search_perf_test tests/search_perf_test.cpp)
target_link_libraries(search_perf_test PRIVATE combat_core)
target_compile_features(search_perf_test PRIVATE cxx_std_23)
target_compile_options(search_perf_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME search_perf_test COMMAND search_perf_test)

# The native value net's inner loops benefit from AVX2/FMA on the machine that runs it.
set_source_files_properties(topology/value_net.cpp PROPERTIES COMPILE_OPTIONS "-march=native")
