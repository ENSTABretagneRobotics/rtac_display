cmake_minimum_required(VERSION 3.10)
project(rtac_display VERSION 1.0)

option(WITH_CUDA "enable cuda related code" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_DOCS  "Builds Doxygen documentation" OFF)

if(${WITH_CUDA})
    if(NOT TARGET rtac_cuda)
        find_package(rtac_cuda 1.0 REQUIRED)
    endif()
endif()


if(NOT TARGET rtac_base)
    find_package(rtac_base 1.0 REQUIRED)
endif()

set(OpenGL_GL_PREFERENCE GLVND)
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

list(APPEND rtac_display_headers
    include/rtac_display/common.h
    include/rtac_display/utils.h
    include/rtac_display/GLState.h
    include/rtac_display/GLContext.h
    include/rtac_display/GLFormat.h
    include/rtac_display/GLSLType.h
    include/rtac_display/Color.h
    include/rtac_display/Scaling.h

    include/rtac_display/views/View.h
    include/rtac_display/views/ImageView.h
    include/rtac_display/views/View3D.h
    include/rtac_display/views/PinholeView.h
    include/rtac_display/views/OrthoView.h
    include/rtac_display/views/PlotView2D.h

    include/rtac_display/renderers/Renderer.h
    include/rtac_display/renderers/Frame.h
    include/rtac_display/renderers/FrameInstances.h
    include/rtac_display/renderers/ImageRenderer.h
    include/rtac_display/renderers/MeshRenderer.h
    include/rtac_display/renderers/FanRenderer.h
    include/rtac_display/renderers/PointCloudRenderer.h
    include/rtac_display/renderers/SimplePlot.h
    include/rtac_display/renderers/Scatter2D.h
    include/rtac_display/renderers/SimplePlot2D.h

    include/rtac_display/Colormap.h
    include/rtac_display/colormaps/Gray.h
    include/rtac_display/colormaps/Viridis.h

    include/rtac_display/DrawingSurface.h
    include/rtac_display/GLFWContext.h
    include/rtac_display/Display.h
    include/rtac_display/GLVector.h
    include/rtac_display/GLTexture.h
    include/rtac_display/GLRenderBuffer.h
    include/rtac_display/GLFrameBuffer.h
    include/rtac_display/GLMesh.h
    include/rtac_display/EventHandler.h
    include/rtac_display/PlotData2D.h

    include/rtac_display/samples/ImageDisplay.h
    include/rtac_display/samples/OrbitControls.h
    include/rtac_display/samples/Display3D.h
    include/rtac_display/samples/PauseHandler.h

    include/rtac_display/GLReductor.h
)
list(APPEND rtac_display_SOURCES
    src/utils.cpp
    src/GLState.cpp
    src/Scaling.cpp

    src/GLMesh.cpp

    src/views/View.cpp
    src/views/ImageView.cpp
    src/views/View3D.cpp
    src/views/PinholeView.cpp
    src/views/OrthoView.cpp
    src/views/PlotView2D.cpp

    src/renderers/Renderer.cpp
    src/renderers/Frame.cpp
    src/renderers/FrameInstances.cpp
    src/renderers/ImageRenderer.cpp
    src/renderers/MeshRenderer.cpp
    src/renderers/FanRenderer.cpp
    src/renderers/SimplePlot.cpp
    src/renderers/Scatter2D.cpp
    src/renderers/SimplePlot2D.cpp

    src/Colormap.cpp
    
    src/DrawingSurface.cpp
    src/Display.cpp
    src/GLTexture.cpp
    src/GLRenderBuffer.cpp
    src/GLFrameBuffer.cpp
    src/EventHandler.cpp
    src/PlotData2D.cpp

    src/samples/ImageDisplay.cpp
    src/samples/OrbitControls.cpp
    src/samples/Display3D.cpp

    src/GLReductor.cpp
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
    list(APPEND rtac_display_headers
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

if(WITH_ASYNC)
    list(APPEND rtac_display_headers
        include/rtac_display/DisplayServer.h
    )
    target_sources(rtac_display PRIVATE src/DisplayServer.cpp)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    rtac_build_docs(rtac_display)
endif()

rtac_install_target(rtac_display
    HEADER_FILES ${rtac_display_headers}
)

# add_subdirectory(bin)
