cmake_minimum_required(VERSION 3.10)
project(optix_display VERSION 0.1)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

if(NOT TARGET optix_helpers)
    find_package(optix_helpers REQUIRED)
endif()

list(APPEND optix_helpers_display_PUBLIC_HEADERS
    include/optix_helpers/display/utils.h
    include/optix_helpers/display/View.h
    include/optix_helpers/display/ImageView.h
    include/optix_helpers/display/View3D.h
    include/optix_helpers/display/Renderer.h
    include/optix_helpers/display/ImageRenderer.h
    include/optix_helpers/display/Display.h
    include/optix_helpers/display/RenderBufferGL.h
)

add_library(optix_helpers_display SHARED
    src/utils.cpp
    src/View.cpp
    src/ImageView.cpp
    src/View3D.cpp
    src/Renderer.cpp
    src/ImageRenderer.cpp
    src/Display.cpp
    src/RenderBufferGL.cpp
)
set_target_properties(optix_helpers_display PROPERTIES 
    PUBLIC_HEADER "${optix_helpers_display_PUBLIC_HEADERS}"
    # INSTALL_RPATH_USE_LINK_PATH True
)
target_include_directories(optix_helpers_display PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(optix_helpers_display PUBLIC
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW
    glfw
    optix_helpers
)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/installation.cmake)
install_target(optix_helpers_display)
