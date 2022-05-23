cmake_minimum_required(VERSION 3.10)
project(rtac_display VERSION 0.1)

option(WITH_CUDA "enable cuda related code" ON)
if(${WITH_CUDA})
    enable_language(CUDA)
    if(NOT TARGET rtac_cuda)
        find_package(rtac_cuda REQUIRED)
    endif()
endif()


if(NOT TARGET rtac_base)
    find_package(rtac_base REQUIRED)
endif()


find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)


# Optionally finding freetype for text rendering
find_package(Freetype)
if(NOT TARGET Freetype::Freetype)
    message(STATUS 
       "WARNING : Could not find freetype library. Text rendering support is disabled."
    )
endif()

list(APPEND rtac_display_PUBLIC_HEADERS
    include/rtac_display/utils.h
    include/rtac_display/GLState.h
    include/rtac_display/GLContext.h
    include/rtac_display/GLFormat.h
    include/rtac_display/GLSLType.h
    include/rtac_display/Color.h

    include/rtac_display/views/View.h
    include/rtac_display/views/ImageView.h
    include/rtac_display/views/View3D.h
    include/rtac_display/views/PinholeView.h
    include/rtac_display/views/OrthoView.h

    include/rtac_display/renderers/Renderer.h
    include/rtac_display/renderers/Frame.h
    include/rtac_display/renderers/FrameInstances.h
    include/rtac_display/renderers/ImageRenderer.h
    include/rtac_display/renderers/PointCloudRenderer.h
    include/rtac_display/renderers/MeshRenderer.h
    include/rtac_display/renderers/NormalsRenderer.h
    include/rtac_display/renderers/TexturedMeshRenderer.h
    include/rtac_display/renderers/FanRenderer.h

    include/rtac_display/Colormap.h
    include/rtac_display/colormaps/Gray.h
    include/rtac_display/colormaps/Viridis.h

    include/rtac_display/DrawingSurface.h
    include/rtac_display/GLFWContext.h
    include/rtac_display/Display.h
    include/rtac_display/GLVector.h
    include/rtac_display/GLTexture.h
    include/rtac_display/GLMesh.h
    include/rtac_display/EventHandler.h

    include/rtac_display/samples/ImageDisplay.h
    include/rtac_display/samples/OrbitControls.h
    include/rtac_display/samples/Display3D.h

    include/rtac_display/GLReductor.h
    include/rtac_display/ObjLoader.h
)
list(APPEND rtac_display_SOURCES
    src/utils.cpp
    src/GLState.cpp

    src/GLMesh.cpp

    src/views/View.cpp
    src/views/ImageView.cpp
    src/views/View3D.cpp
    src/views/PinholeView.cpp
    src/views/OrthoView.cpp

    src/renderers/Renderer.cpp
    src/renderers/Frame.cpp
    src/renderers/FrameInstances.cpp
    src/renderers/ImageRenderer.cpp
    src/renderers/PointCloudRenderer.cpp
    src/renderers/MeshRenderer.cpp
    src/renderers/NormalsRenderer.cpp
    src/renderers/FanRenderer.cpp

    src/Colormap.cpp
    
    src/DrawingSurface.cpp
    src/Display.cpp
    src/GLTexture.cpp
    src/EventHandler.cpp

    src/samples/ImageDisplay.cpp
    src/samples/OrbitControls.cpp
    src/samples/Display3D.cpp

    src/GLReductor.cpp
    src/ObjLoader.cpp
)
add_library(rtac_display SHARED ${rtac_display_SOURCES})
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
    target_link_libraries(rtac_display PUBLIC rtac_cuda)
    target_compile_definitions(rtac_display PUBLIC RTAC_DISPLAY_CUDA)
    list(APPEND CONFIG_COMMANDS "find_package(rtac_cuda REQUIRED)")
endif()

if(TARGET Freetype::Freetype)
    list(APPEND rtac_display_PUBLIC_HEADERS
        include/rtac_display/text/freetype.h
        include/rtac_display/text/Glyph.h
        include/rtac_display/text/FontFace.h
        include/rtac_display/text/TextRenderer.h
    )
    target_sources(rtac_display PRIVATE 
        src/text/freetype.cpp
        src/text/Glyph.cpp
        src/text/FontFace.cpp
        src/text/TextRenderer.cpp
    )
    target_link_libraries(rtac_display PUBLIC Freetype::Freetype)
endif()

add_subdirectory(tests)

set_target_properties(rtac_display PROPERTIES 
    RTAC_PUBLIC_HEADERS "${rtac_display_PUBLIC_HEADERS}"
    INSTALL_RPATH_USE_LINK_PATH True
)
rtac_install_target(rtac_display
    ADDITIONAL_CONFIG_COMMANDS ${CONFIG_COMMANDS}
)

# add_subdirectory(bin)
