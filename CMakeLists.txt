cmake_minimum_required(VERSION 3.10)
project(rtac_display VERSION 0.1)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

find_package(rtac_base REQUIRED)

list(APPEND rtac_display_PUBLIC_HEADERS
    include/rtac_display/Handle.h
    include/rtac_display/utils.h
    include/rtac_display/views/View.h
    include/rtac_display/views/ImageView.h
    include/rtac_display/views/View3D.h
    include/rtac_display/views/PinholeView.h
    include/rtac_display/views/OrthoView.h
    include/rtac_display/renderers/Renderer.h
    include/rtac_display/renderers/ImageRenderer.h
    include/rtac_display/renderers/PointCloudRenderer.h
    include/rtac_display/renderers/MeshRenderer.h
    include/rtac_display/renderers/NormalsRenderer.h
    include/rtac_display/Display.h
    include/rtac_display/GLVector.h
)

add_library(rtac_display SHARED
    src/utils.cpp
    src/views/View.cpp
    src/views/ImageView.cpp
    src/views/View3D.cpp
    src/views/PinholeView.cpp
    src/views/OrthoView.cpp
    src/renderers/Renderer.cpp
    src/renderers/ImageRenderer.cpp
    src/renderers/PointCloudRenderer.cpp
    src/renderers/MeshRenderer.cpp
    src/renderers/NormalsRenderer.cpp
    src/Display.cpp
)
set_target_properties(rtac_display PROPERTIES 
    PUBLIC_HEADER "${rtac_display_PUBLIC_HEADERS}"
    INSTALL_RPATH_USE_LINK_PATH True
)
target_include_directories(rtac_display PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(rtac_display PUBLIC
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW
    glfw
    rtac_base
)

add_subdirectory(tests)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/installation.cmake)
install_target(rtac_display)
