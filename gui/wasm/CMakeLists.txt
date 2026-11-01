cmake_minimum_required(VERSION 3.25)
project(gumbel_browser LANGUAGES CXX)
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Configure with emcmake cmake")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(STS_LIGHTSPEED_DIR "${ROOT}/../lightspeed" CACHE PATH "sts_lightspeed checkout")
# The simulator's old vendored JSON assumes std::array iterators are raw pointers.
# A pinned modern header supports libc++ without patching the shared simulator checkout.
set(JSON_HEADER "${CMAKE_CURRENT_BINARY_DIR}/json/nlohmann/json.hpp")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/json/nlohmann")
if(NOT EXISTS "${JSON_HEADER}")
    file(DOWNLOAD "https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp"
        "${JSON_HEADER}" EXPECTED_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63 TLS_VERIFY ON)
endif()
file(SHA256 "${JSON_HEADER}" JSON_HASH)
if(NOT JSON_HASH STREQUAL "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63")
    message(FATAL_ERROR "Unexpected browser JSON header checksum")
endif()
add_subdirectory("${STS_LIGHTSPEED_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/lightspeed")
target_include_directories(sts_lightspeed SYSTEM BEFORE PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/json")
add_executable(gumbel
    bridge.cpp
    ${ROOT}/agents/combat/nn/decision.cpp
    ${ROOT}/agents/combat/nn/explanation.cpp
    ${ROOT}/agents/combat/nn/tokens.cpp
    ${ROOT}/agents/combat/search/gumbel/gumbel.cpp
    ${ROOT}/agents/combat/search/rollout_blend/rollout_blend.cpp
    ${ROOT}/environments/combat/public.cpp
    ${ROOT}/environments/combat/view.cpp
    ${ROOT}/environments/combat/environment.cpp
    ${ROOT}/environments/combat/encoding_v4.cpp
    ${ROOT}/environments/combat/record_v4.cpp)
target_include_directories(gumbel PRIVATE "${ROOT}")
target_link_libraries(gumbel PRIVATE sts_lightspeed)
# Both our code and simulator must retain C++ exception handling.
# The simulator relies on libstdc++ transitive includes absent from Emscripten's libc++.
# Keep this workaround local rather than modifying the shared checkout.
target_compile_options(sts_lightspeed PRIVATE -fexceptions -include algorithm)
target_compile_options(gumbel PRIVATE -fexceptions)
target_link_options(gumbel PRIVATE
    -fexceptions --no-entry
    "-sMODULARIZE=1" "-sEXPORT_NAME=createGumbel" "-sENVIRONMENT=worker"
    "-sEXPORTED_FUNCTIONS=['_explain_json']" "-sEXPORTED_RUNTIME_METHODS=['ccall']"
    "-sASYNCIFY=1" "-sASYNCIFY_STACK_SIZE=1048576"
    "-sALLOW_MEMORY_GROWTH=1" "-sSTACK_SIZE=16777216" "-sINITIAL_MEMORY=67108864")
