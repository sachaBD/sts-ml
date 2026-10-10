# Isolated build of pv_continuation (diagnostic app) without editing the shared root CMakeLists:
#   cmake -S apps/continuation -B build/continuation-v1 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/continuation-v1 --target pv_continuation -j2
# Reuses the repository targets (combat_core, pv_onnx, sts_lightspeed) with the root project's settings (LTO on).
cmake_minimum_required(VERSION 3.25)
project(pv_continuation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." stsrl)
if(NOT TARGET pv_onnx)
    message(FATAL_ERROR "pv_onnx unavailable (ONNX Runtime / pyarrow setup)")
endif()
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python" -c "import pyarrow; print(pyarrow.__path__[0], end='')"
    OUTPUT_VARIABLE CONTINUATION_PYARROW_DIR ERROR_QUIET)
file(GLOB CONTINUATION_ARROW_LIB "${CONTINUATION_PYARROW_DIR}/libarrow.so.*")
file(GLOB CONTINUATION_PARQUET_LIB "${CONTINUATION_PYARROW_DIR}/libparquet.so.*")
if(NOT CONTINUATION_ARROW_LIB OR NOT CONTINUATION_PARQUET_LIB)
    message(FATAL_ERROR "pyarrow Arrow/Parquet unavailable for continuation shard writer")
endif()
add_executable(pv_continuation "${CMAKE_CURRENT_SOURCE_DIR}/continuation.cpp")
target_include_directories(pv_continuation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_include_directories(pv_continuation SYSTEM PRIVATE "${CONTINUATION_PYARROW_DIR}/include")
target_link_libraries(pv_continuation PRIVATE pv_onnx ${CONTINUATION_PARQUET_LIB} ${CONTINUATION_ARROW_LIB} OpenSSL::Crypto)
target_link_options(pv_continuation PRIVATE "-Wl,-rpath,${CONTINUATION_PYARROW_DIR}")
target_compile_options(pv_continuation PRIVATE -Wall -Wextra -Wpedantic)
