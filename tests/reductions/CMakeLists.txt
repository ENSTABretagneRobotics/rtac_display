

set(target_name reductions_${PROJECT_NAME})
add_executable(${target_name}
    src/reductions.cu
    src/reductions.cpp
    src/main.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_display
    rtac_cuda
)
if(WITH_CUDA)
    set_target_properties(${target_name} PROPERTIES
                          CUDA_ARCHITECTURES native)
endif()
