cmake_minimum_required( VERSION 2.8 )
project( libkotekan_hsa )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)

include_directories ( /opt/rocm/hsa/include/ )

include ( hsa.cmake )

add_library ( kotekan_hsa
              ${KOTEKAN_HSA_SOURCES})

target_link_libraries( kotekan_hsa kotekan_utils )

add_subdirectory(kernels)
add_dependencies( kotekan_hsa makeHSACOKernels )




