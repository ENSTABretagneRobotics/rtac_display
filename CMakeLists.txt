cmake_minimum_required(VERSION 3.10)
project(optix_display VERSION 0.1)

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

if(NOT TARGET optix_helpers)
    find_package(optix_helpers REQUIRED)
endif()

list(APPEND optix_helpers_display_PUBLIC_HEADERS
    include/optix_helpers/display/Display.h
    include/optix_helpers/display/RenderBufferGL.h
)

add_library(optix_helpers_display SHARED
    src/utils.cpp
    src/Display.cpp
    src/RenderBufferGL.cpp
)
target_include_directories(optix_helpers_display PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:/include>
)
target_link_libraries(optix_helpers_display PUBLIC
    glfw
    OpenGL::GL
    GLEW::GLEW
    optix_helpers
)

add_subdirectory(tests)
