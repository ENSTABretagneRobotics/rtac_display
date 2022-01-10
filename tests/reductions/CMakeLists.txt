

set(target_name reductions_${PROJECT_NAME})
add_executable(${target_name}
    src/main.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_display
)
