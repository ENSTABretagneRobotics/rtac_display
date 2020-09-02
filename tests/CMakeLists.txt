project(optix_display_test VERSION 0.1)

add_executable(display_test
    src/main.cpp
)
target_link_libraries(display_test
    optix_helpers_display
)
