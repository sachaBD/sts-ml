# Actors + InferenceServer: no ONNX Runtime or CUDA needed.
add_library(infra_actors STATIC actors.cpp)
target_include_directories(infra_actors PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_options(infra_actors PRIVATE -Wall -Wextra -Wpedantic)

add_executable(infra_actors_test actors_test.cpp)
target_link_libraries(infra_actors_test PRIVATE infra_actors)
target_compile_options(infra_actors_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME infra_actors_test COMMAND infra_actors_test)
set_tests_properties(infra_actors_test PROPERTIES TIMEOUT 30)
