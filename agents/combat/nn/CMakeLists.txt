# Combat owns episodes/search/encoding; infra owns workers and inference.
if(NOT TARGET infra_inference)
    message(STATUS "nn_play not built: infra_inference unavailable")
    return()
endif()
add_executable(nn_play play.cpp tokens.cpp)
target_link_libraries(nn_play PRIVATE combat_core combat_search infra_actors infra_inference)
target_compile_options(nn_play PRIVATE -Wall -Wextra -Wpedantic)
