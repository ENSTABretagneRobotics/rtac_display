

find_package(CLI11)
if(TARGET CLI11::CLI11)
    add_executable(rtac_ply_viewer
        src/ply_viewer.cpp
    )
    target_link_libraries(rtac_ply_viewer PRIVATE
        rtac_display
        CLI11::CLI11
    )
else()
    message(WARNING "Could not find CLI11 library. rtac_ply_viewer won't be built.")
endif()
