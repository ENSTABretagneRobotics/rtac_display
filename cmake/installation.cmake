function(install_target TARGET_NAME)

	set(multiValueArgs ADDITIONNAL_CONFIG_FILES)
	cmake_parse_arguments(INSTALLATION "${options}" "${oneValueArgs}"
	                      "${multiValueArgs}" ${ARGN} )

	include(GNUInstallDirs)
	# Configuration
	set(VERSION_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}ConfigVersion.cmake")
	set(PROJECT_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}Config.cmake")
	set(TARGET_EXPORT_NAME "${TARGET_NAME}Targets")
	set(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME}")
	
	include(CMakePackageConfigHelpers)
	write_basic_package_version_file(
	    "${VERSION_CONFIG}" COMPATIBILITY SameMajorVersion
	)
	configure_package_config_file(
	    "cmake/Config.cmake.in"
	    "${PROJECT_CONFIG}"
	    INSTALL_DESTINATION "${CONFIG_INSTALL_DIR}"
	    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
	)
    # get_target_property(HEADERS ${TARGET_NAME} PUBLIC_HEADER)
    # message(STATUS "Public headers : ${HEADERS}")

	
	# Installation
	install(
	    TARGETS "${TARGET_NAME}"
	    EXPORT "${TARGET_EXPORT_NAME}"
	    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	    # PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${INSTALLATION_HEADER_PREFIX}"
	)
    list(APPEND CONFIG_FILES_TO_INSTALL
        ${PROJECT_CONFIG}
        ${VERSION_CONFIG}
        ${INSTALLATION_ADDITIONNAL_CONFIG_FILES}
    )
	install(
	    FILES ${CONFIG_FILES_TO_INSTALL}
	    DESTINATION "${CONFIG_INSTALL_DIR}"
	)

    # Installing header files
    get_target_property(TARGET_TYPE_VALUE ${TARGET_NAME} TYPE)
    if(${TARGET_TYPE_VALUE} STREQUAL "INTERFACE_LIBRARY")
        get_target_property(HEADER_FILES ${TARGET_NAME} INTERFACE_PUBLIC_HEADER)
    else()
        get_target_property(HEADER_FILES ${TARGET_NAME} PUBLIC_HEADER)
    endif()
    if(NOT "${HEADER_FILES}" STREQUAL "HEADER_FILES-NOTFOUND")
        foreach(header ${HEADER_FILES})
            get_filename_component(header_dir ${header} DIRECTORY)
            install(FILES ${header}
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/../${header_dir}")
        endforeach()
    endif()

    # Export ting target ?
	install(
	    EXPORT "${TARGET_EXPORT_NAME}"
	    # NAMESPACE "${PROJECT_NAME}::"
	    DESTINATION "${CONFIG_INSTALL_DIR}"
	)

endfunction()

