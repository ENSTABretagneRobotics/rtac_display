cmake_minimum_required(VERSION 3.10)
project(optix_display VERSION 0.1)

find_package(glfw3 REQUIRED)

add_library(optix_helpers_display SHARED
    src/Display.cpp
)
target_include_directories(optix_helpers_display PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:/include>
)
target_link_libraries(optix_helpers_display PUBLIC
   glfw
)

add_subdirectory(tests)
