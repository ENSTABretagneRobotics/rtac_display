
list(APPEND test_files
    src/base_test.cpp
    src/view_test.cpp
    src/mesh_test.cpp
    src/texture_test.cpp
    src/texturedmesh_test.cpp
    src/pointcloud_test.cpp
    src/imagedisplay_test.cpp
    src/userinput_test.cpp
    src/glfwinput_test.cpp
    src/dual_window.cpp
    src/gl_mesh_test.cpp
    src/glsl_types.cpp
    src/fan_renderer.cpp
    src/instances_renderer.cpp
    src/png_codec.cpp
    src/obj_loader.cpp
    src/display_server.cpp
)

foreach(filename ${test_files})
    get_filename_component(test_name ${filename} NAME_WE)
    set(target_name ${test_name}_${PROJECT_NAME})
    add_executable(${target_name} ${filename})
    target_link_libraries(${target_name} rtac_display)
    if(WITH_CUDA)
        set_target_properties(${target_name} PROPERTIES
                              CUDA_ARCHITECTURES native)
    endif()
endforeach()

if(TARGET Freetype::Freetype)
    add_subdirectory(freetype_test)
endif()

add_subdirectory(reductions)


