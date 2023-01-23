


find_package(CLI11)
if(TARGET CLI11::CLI11)
    add_executable(rtac_obj_viewer
        src/obj_viewer.cpp
    )
    target_link_libraries(rtac_obj_viewer PRIVATE rtac_display)
else()
    message(WARNING "Could not find CLI11 library. rtac_obj_viewer won't be built.")
endif()
