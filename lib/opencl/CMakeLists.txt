cmake_minimum_required( VERSION 2.8 )
project( kotekan_opencl )

include_directories(${OpenCL_INCLUDE_DIR})

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

include ( opencl.cmake )

add_library ( kotekan_opencl KOTEKAN_OPENCL_SOURCES )

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "kernels/")

add_dependencies( kotekan_opencl kernel_copy kotekan_gpu )
