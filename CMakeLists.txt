cmake_minimum_required(VERSION 3.25)

project(sts_combat_rl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link-time optimization for everything below, sts_lightspeed included: its many small out-of-line accessors
# (CardInstance / Monster getters) get inlined across files: ~1.47x search throughput, identical results.
# Opt out with -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "LTO not supported, building without it: ${ipo_error}")
    endif()
endif()

# The simulator: the sibling ../lightspeed checkout.
set(STS_LIGHTSPEED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lightspeed" CACHE PATH "sts_lightspeed checkout")
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sts_lightspeed")

# The rules of a fight as a player sees them, and building a fight from its saved start.
add_library(combat_environment STATIC environments/combat/public.cpp environments/combat/fight_start.cpp)
target_include_directories(combat_environment PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(combat_environment PUBLIC sts_lightspeed)
target_compile_options(combat_environment PRIVATE -Wall -Wextra -Wpedantic)

# Fixed fights used only by tests.
add_library(combat_scenarios STATIC environments/combat/scenarios/jaw_worm.cpp)
target_link_libraries(combat_scenarios PUBLIC sts_lightspeed)
target_include_directories(combat_scenarios PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

enable_testing()
# GoogleTest for native unit tests, pinned and built with the project (no system install needed).
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# toml++ reads the TOML files the native code is given (nn_play's settings, the exported model directory), pinned.
FetchContent_Declare(tomlplusplus URL https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(tomlplusplus)


# Colocated Python tests share CTest with native checks. Never recurse through runs/ output trees.
set(STSRL_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3" CACHE FILEPATH "Python interpreter for tests")
file(GLOB_RECURSE python_test_files CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "agents/test_*.py" "infra/test_*.py" "environments/test_*.py" "apps/test_*.py")
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

# Workers depend on inference, never the reverse.
add_subdirectory(infra/inference)
add_subdirectory(infra/workers)
add_subdirectory(agents/combat/search)
add_subdirectory(agents/combat/nn)
add_subdirectory(agents/combat/player)
