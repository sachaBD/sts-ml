# Backends turn tokens into predictions; they know nothing about threads or workers.
# ONNX Runtime (GPU build) from the venv's onnxruntime-gpu wheel, whose CUDA/TensorRT provider
# libraries must sit next to libonnxruntime. Headers are prepared in build/deps/onnxruntime/include.
execute_process(COMMAND "${STSRL_PYTHON}" -c "import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).parent / 'capi', end='')"
    OUTPUT_VARIABLE INFRA_ORT_LIB ERROR_QUIET)
set(INFRA_ORT_INCLUDE "${PROJECT_SOURCE_DIR}/build/deps/onnxruntime/include")
file(GLOB INFRA_ORT_SO "${INFRA_ORT_LIB}/libonnxruntime.so.*")
find_package(CUDAToolkit QUIET)
if(NOT EXISTS "${INFRA_ORT_INCLUDE}/onnxruntime_cxx_api.h" OR NOT INFRA_ORT_SO OR NOT CUDAToolkit_FOUND)
    message(STATUS "infra/inference not built: need ONNX Runtime headers/wheel and the CUDA toolkit")
    return()
endif()
# The wheel ships a versioned library without its SONAME link: keep library and providers together.
set(INFRA_ORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime")
file(MAKE_DIRECTORY "${INFRA_ORT_DIR}")
file(CREATE_LINK "${INFRA_ORT_SO}" "${INFRA_ORT_DIR}/libonnxruntime.so.1" SYMBOLIC)
file(GLOB INFRA_ORT_PROVIDERS "${INFRA_ORT_LIB}/libonnxruntime_providers_*.so")
foreach(provider ${INFRA_ORT_PROVIDERS})
    get_filename_component(name "${provider}" NAME)
    file(CREATE_LINK "${provider}" "${INFRA_ORT_DIR}/${name}" SYMBOLIC)
endforeach()

add_library(infra_inference STATIC cpu.cpp gpu.cpp)
target_include_directories(infra_inference PUBLIC "${PROJECT_SOURCE_DIR}")
target_include_directories(infra_inference SYSTEM PRIVATE "${INFRA_ORT_INCLUDE}")
target_link_libraries(infra_inference PUBLIC "${INFRA_ORT_DIR}/libonnxruntime.so.1" CUDA::cudart)
target_link_options(infra_inference PUBLIC "-Wl,-rpath,${INFRA_ORT_DIR}")
target_compile_options(infra_inference PRIVATE -Wall -Wextra -Wpedantic)
