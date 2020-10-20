cmake_minimum_required(VERSION 3.10)
project(rtac_display VERSION 0.1)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

find_package(rtac_base REQUIRED)

list(APPEND rtac_display_PUBLIC_HEADERS
    include/rtac_display/Handle.h
    include/rtac_display/utils.h
    include/rtac_display/View.h
    include/rtac_display/ImageView.h
    include/rtac_display/View3D.h
    include/rtac_display/PinholeView.h
    include/rtac_display/OrthoView.h
    include/rtac_display/Renderer.h
    include/rtac_display/ImageRenderer.h
    include/rtac_display/PointCloudRenderer.h
    include/rtac_display/MeshRenderer.h
    include/rtac_display/NormalsRenderer.h
    include/rtac_display/Display.h
)

add_library(rtac_display SHARED
    src/utils.cpp
    src/View.cpp
    src/ImageView.cpp
    src/View3D.cpp
    src/PinholeView.cpp
    src/OrthoView.cpp
    src/Renderer.cpp
    src/ImageRenderer.cpp
    src/PointCloudRenderer.cpp
    src/MeshRenderer.cpp
    src/NormalsRenderer.cpp
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
