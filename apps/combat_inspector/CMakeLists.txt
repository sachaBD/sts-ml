add_library(combat_inspection STATIC inspection.cpp)
target_link_libraries(combat_inspection PUBLIC player_decision)
target_compile_options(combat_inspection PRIVATE -Wall -Wextra -Wpedantic)

if(TARGET infra_inference)
    add_executable(combat_explain explain.cpp)
    target_link_libraries(combat_explain PRIVATE combat_inspection nn_network)
    target_compile_options(combat_explain PRIVATE -Wall -Wextra -Wpedantic)
endif()
