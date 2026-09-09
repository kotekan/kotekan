project(kotekan)

if(HIP_FOUND)
    hip_add_executable(kotekan kotekan.cpp)
else()
    add_executable(kotekan kotekan.cpp)
endif()

target_compile_definitions(kotekan PRIVATE mssse3 __STDC_LIMIT_MACROS)

# If HDF5 is enabled and we detected a plugin directory, pass it to the binary so it can set the
# path programatically at runtime.
if(HDF5_FOUND
   AND HighFive_FOUND
   AND KOTEKAN_HDF5_PLUGIN_DIR)
    target_include_directories(kotekan SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
    # Escape any quotes or spaces as a C string literal
    string(REPLACE "\"" "\\\"" _HDF5_PLUGIN_DIR_ESC "${KOTEKAN_HDF5_PLUGIN_DIR}")
    target_compile_definitions(kotekan PRIVATE DEFAULT_HDF5_PLUGIN_PATH="${_HDF5_PLUGIN_DIR_ESC}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(
        kotekan
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    if(ASDF_CXX_FOUND)
        target_link_libraries(kotekan PRIVATE blosc)
        # asdf-cxx's pkg-config declares -lblosc (c-blosc 1.x), but an install built against
        # c-blosc2 references blosc2_* from its static archive as well, and the .pc file does not
        # say which. Link blosc2 too where it exists: --no-as-needed because the reference comes
        # from an archive the linker has not seen yet at this point in the line (the .so would be
        # dropped as unused); push/pop-state so the flag does not stick to the libraries that
        # follow; the resolved path rather than -lblosc2 so CMake's prefixes are searched.
        find_library(BLOSC2_LIB blosc2)
        if(BLOSC2_LIB)
            target_link_libraries(kotekan PRIVATE "-Wl,--push-state,--no-as-needed" ${BLOSC2_LIB}
                                                  "-Wl,--pop-state")
        endif()
    endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(kotekan PRIVATE -Wl,-all_load kotekan_libs)
else()
    message(FATAL_ERROR "System type ${CMAKE_SYSTEM_NAME} not supported")
endif()

if(Julia_FOUND)
    # Keep libjulia on the runtime search path when Julia is installed outside standard locations.
    set_property(
        TARGET kotekan
        APPEND
        PROPERTY BUILD_RPATH "${JULIA_LIBRARY_DIR}")
    set_property(
        TARGET kotekan
        APPEND
        PROPERTY INSTALL_RPATH "${JULIA_LIBRARY_DIR}")
endif()

if(TARGET n2k)
    # Keep vendor libn2k discoverable from build and install locations.
    set_property(
        TARGET kotekan
        APPEND
        PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:n2k>")
    set_property(
        TARGET kotekan
        APPEND
        PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

install(
    TARGETS kotekan
    DESTINATION /usr/local/bin
    COMPONENT binaries)
