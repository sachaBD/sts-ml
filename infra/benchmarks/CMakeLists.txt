# Component benchmarks; run them all with `python -m infra.benchmarks` (see __main__.py).
add_executable(infra_bench_actors actors.cpp)
target_link_libraries(infra_bench_actors PRIVATE infra_actors)
target_compile_options(infra_bench_actors PRIVATE -Wall -Wextra -Wpedantic)

if(TARGET infra_inference)
    add_executable(infra_bench_inference inference.cpp)
    target_link_libraries(infra_bench_inference PRIVATE infra_inference)
    target_compile_options(infra_bench_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()
