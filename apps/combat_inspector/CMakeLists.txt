if(TARGET infra_inference)
    add_library(combat_inspection STATIC inspection.cpp)
    target_link_libraries(combat_inspection PUBLIC player_decision)
    target_compile_options(combat_inspection PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(combat_explain explain.cpp)
    target_link_libraries(combat_explain PRIVATE combat_inspection nn_network infra_actors CLI11::CLI11)
    target_compile_options(combat_explain PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(combat_lab lab.cpp)
    target_link_libraries(combat_lab PRIVATE combat_inspection nn_network CLI11::CLI11)
    target_compile_options(combat_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()
