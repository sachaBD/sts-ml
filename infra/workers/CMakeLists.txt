# Actors + InferenceServer: no ONNX Runtime or CUDA needed.
add_library(infra_inference_service STATIC inference_service.cpp)
target_include_directories(infra_inference_service PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_options(infra_inference_service PRIVATE -Wall -Wextra -Wpedantic)

add_library(infra_actors STATIC actors.cpp)
target_link_libraries(infra_actors PUBLIC infra_inference_service)
target_include_directories(infra_actors PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_options(infra_actors PRIVATE -Wall -Wextra -Wpedantic)

add_executable(infra_actors_test actors_test.cpp)
target_link_libraries(infra_actors_test PRIVATE infra_actors GTest::gtest_main)
target_compile_options(infra_actors_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(infra_actors_test PROPERTIES TIMEOUT 30)

add_executable(inference_service_test inference_service_test.cpp)
target_link_libraries(inference_service_test PRIVATE infra_inference_service GTest::gtest_main)
target_compile_options(inference_service_test PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(inference_service_test PROPERTIES TIMEOUT 30)
