project(kotekan)

if(${USE_HIP})
    hip_add_executable(kotekan kotekan.cpp)
else()
    add_executable(kotekan kotekan.cpp)
endif()

target_compile_definitions(kotekan PRIVATE mssse3 __STDC_LIMIT_MACROS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(
        kotekan
        PRIVATE -Wl,--whole-archive $<TARGET_PROPERTY:kotekan_libs,INTERFACE_LINK_LIBRARIES>
                -Wl,--no-whole-archive)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(kotekan PRIVATE -Wl,-all_load kotekan_libs)
else()
    message(FATAL_ERROR "System type ${CMAKE_SYSTEM_NAME} not supported")
endif()

set_property(TARGET kotekan PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install(
    TARGETS kotekan
    DESTINATION /usr/local/bin
    COMPONENT binaries)
