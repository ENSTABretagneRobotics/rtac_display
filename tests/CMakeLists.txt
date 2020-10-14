project(optix_display_test VERSION 0.1)

list(APPEND test_names
    base_test
    view_test
    animation_test
    render3d_test
    mesh_test
    pointcloud_test
)
list(APPEND test_deps
    optix_helpers
    optix_helpers_display
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach()
