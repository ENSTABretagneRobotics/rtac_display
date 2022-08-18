

set(target_name freetype_test_${PROJECT_NAME})
add_executable(${target_name}
    src/main.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_display
)
if(WITH_CUDA)
    set_target_properties(${target_name} PROPERTIES
                          CUDA_ARCHITECTURES native)
endif()
