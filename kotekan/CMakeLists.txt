project(kotekan)

if(${USE_HIP})
    hip_add_executable(kotekan kotekan.cpp)
else()
    add_executable(kotekan kotekan.cpp)
endif()

target_compile_definitions(kotekan PRIVATE mssse3 __STDC_LIMIT_MACROS)

# If HDF5 is enabled and we detected a plugin directory, pass it to the
# binary so it can set HDF5_PLUGIN_PATH at runtime when unset.
if(${USE_HDF5} AND KOTEKAN_HDF5_PLUGIN_DIR)
    # Escape any quotes or spaces as a C string literal
    string(REPLACE "\"" "\\\"" _HDF5_PLUGIN_DIR_ESC "${KOTEKAN_HDF5_PLUGIN_DIR}")
    target_compile_definitions(kotekan PRIVATE DEFAULT_HDF5_PLUGIN_PATH="${_HDF5_PLUGIN_DIR_ESC}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(
        kotekan
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
    if(${USE_ASDF})
        target_link_libraries(kotekan PRIVATE blosc)
    endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(kotekan PRIVATE -Wl,-all_load kotekan_libs)
else()
    message(FATAL_ERROR "System type ${CMAKE_SYSTEM_NAME} not supported")
endif()

install(
    TARGETS kotekan
    DESTINATION /usr/local/bin
    COMPONENT binaries)
