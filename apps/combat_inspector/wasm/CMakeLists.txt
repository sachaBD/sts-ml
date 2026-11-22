cmake_minimum_required(VERSION 3.25)
project(combat_browser LANGUAGES CXX)
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Configure with emcmake cmake")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(STS_LIGHTSPEED_DIR "${ROOT}/../lightspeed" CACHE PATH "sts_lightspeed checkout")
set(STSRL_PYTHON "${ROOT}/.venv/bin/python" CACHE FILEPATH "Python for layout generation")
# The simulator's vendored JSON assumes std::array iterators are pointers; libc++ needs a newer header.
set(JSON_HEADER "${CMAKE_CURRENT_BINARY_DIR}/json/nlohmann/json.hpp")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/json/nlohmann")
file(DOWNLOAD "https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp"
    "${JSON_HEADER}" EXPECTED_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63 TLS_VERIFY ON)
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/lightspeed")
target_include_directories(sts_lightspeed SYSTEM BEFORE PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/json")
set(LAYOUT "${CMAKE_CURRENT_BINARY_DIR}/generated/agents/combat/nn/layout.hpp")
add_custom_command(OUTPUT "${LAYOUT}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/agents/combat/nn"
    COMMAND "${STSRL_PYTHON}" "${ROOT}/agents/combat/nn/layout.py" "${LAYOUT}"
    DEPENDS "${ROOT}/agents/combat/nn/layout.py" "${ROOT}/agents/combat/nn/layout.toml")
add_executable(gumbel bridge.cpp "${LAYOUT}"
    ${ROOT}/apps/combat_inspector/inspection.cpp
    ${ROOT}/agents/combat/player/decision.cpp
    ${ROOT}/agents/combat/player/targets.cpp
    ${ROOT}/agents/combat/nn/features.cpp
    ${ROOT}/agents/combat/nn/tokens.cpp
    ${ROOT}/agents/combat/nn/entity/encoding.cpp
    ${ROOT}/agents/combat/nn/inventory/encoding.cpp
    ${ROOT}/agents/combat/search/config.cpp
    ${ROOT}/agents/combat/search/gumbel/gumbel.cpp
    ${ROOT}/agents/combat/search/rollout_blend/rollout_blend.cpp
    ${ROOT}/environments/combat/public.cpp
    ${ROOT}/environments/combat/view.cpp
    ${ROOT}/environments/combat/fight_start.cpp)
target_include_directories(gumbel PRIVATE "${ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(gumbel PRIVATE sts_lightspeed)
# Keep exception handling and libc++ transitive-include workarounds local to this build.
target_compile_options(sts_lightspeed PRIVATE -fexceptions -include algorithm)
# Emscripten's libc++ still gates std::osyncstream behind this flag.
target_compile_options(gumbel PRIVATE -fexceptions -fexperimental-library)
target_link_options(gumbel PRIVATE -fexceptions --no-entry
    "-sMODULARIZE=1" "-sEXPORT_NAME=createGumbel" "-sENVIRONMENT=worker"
    "-sEXPORTED_FUNCTIONS=['_explain_json']" "-sEXPORTED_RUNTIME_METHODS=['ccall']"
    "-sASYNCIFY=1" "-sASYNCIFY_STACK_SIZE=1048576"
    "-sALLOW_MEMORY_GROWTH=1" "-sSTACK_SIZE=16777216" "-sINITIAL_MEMORY=67108864")
