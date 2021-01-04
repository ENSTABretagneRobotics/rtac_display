project(optix_display_test VERSION 0.1)

list(APPEND test_names
    base_test
    view_test
    mesh_test
    pointcloud_test
    normals_test
    glvector_test
)
list(APPEND test_deps
    rtac_display
)

foreach(name ${test_names})
    add_executable(${PROJECT_NAME}_${name} src/${name}.cpp)
    target_link_libraries(${PROJECT_NAME}_${name} ${test_deps})
endforeach()
