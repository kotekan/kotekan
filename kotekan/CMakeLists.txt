cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp kotekanMode.cpp)


include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories (${KOTEKAN_SOURCE_DIR}/include)

link_directories (${KOTEKAN_BINARY_DIR}/lib)

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)

add_executable(kotekan ${SOURCE_EXE})

target_link_libraries( kotekan m dl pthread )
target_link_libraries( kotekan kotekan_libs )
target_link_libraries( kotekan libinclude )

if (${USE_OPENCL})
    add_custom_target(kernel_opencl_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "opencl/kernels/")
    add_dependencies( kotekan kernel_opencl_copy )
    install(FILES kotekan_opencl_280x.yaml DESTINATION /etc/kotekan/ COMPONENT config)
    install(FILES kotekan_opencl_fury.yaml DESTINATION /etc/kotekan/ COMPONENT config)
    install(FILES kotekan_opencl_test.yaml DESTINATION /etc/kotekan/ COMPONENT config)
endif ()

install(TARGETS kotekan DESTINATION /usr/local/bin COMPONENT binaries)

add_definitions(/D__STDC_FORMAT_MACROS)
