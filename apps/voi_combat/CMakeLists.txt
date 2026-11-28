if(TARGET nn_network)
    add_executable(voi_combat_bridge bridge.cpp)
    target_link_libraries(voi_combat_bridge PRIVATE player_decision nn_network)
endif()
