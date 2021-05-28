

find_package(freetype REQUIRED)

set(target_name ${PROJECT_NAME}_freetype_test)
add_executable(${target_name}
    src/main.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_display
    freetype
)
