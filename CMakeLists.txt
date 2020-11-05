cmake_minimum_required(VERSION 3.10)
project(rtac_display VERSION 0.1)

option(WITH_CUDA "enable cuda related code" ON)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

find_package(rtac_base REQUIRED)

list(APPEND rtac_display_PUBLIC_HEADERS
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
list(APPEND rtac_display_SOURCES
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
if(${WITH_CUDA})
    enable_language(CUDA)
    find_package(rtac_cuda REQUIRED)
    list(APPEND rtac_display_PUBLIC_HEADERS
        include/rtac_display/cuda/utils.h
    )
    list(APPEND rtac_display_SOURCES
        src/cuda/utils.cu
    )
endif()
add_library(rtac_display SHARED ${rtac_display_SOURCES})
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
if(${WITH_CUDA})
    target_compile_options(rtac_display PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>
    )
    target_link_libraries(rtac_display PUBLIC rtac_cuda)
    list(APPEND CONFIG_COMMANDS "find_package(rtac_cuda REQUIRED)")
endif()


add_subdirectory(tests)

rtac_install_target(rtac_display
    ADDITIONAL_CONFIG_COMMANDS ${CONFIG_COMMANDS}
)
