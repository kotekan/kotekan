cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/metadata)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/dpdk)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

add_library ( kotekan_libs "empty.c" )

add_subdirectory(core)
target_link_libraries( kotekan_libs kotekan_core )

add_subdirectory(metadata)
target_link_libraries( kotekan_libs kotekan_metadata )

add_subdirectory(utils)
target_link_libraries( kotekan_libs kotekan_utils )

### SELF-REGISTERING BITS:
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries( kotekan_libs -Wl,--whole-archive )
endif ()
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries( kotekan_libs -Wl,-all_load )
endif ()


add_subdirectory(testing)
target_link_libraries( kotekan_libs kotekan_testing )

add_subdirectory(processes)
target_link_libraries( kotekan_libs kotekan_processes )

if (${USE_HCC})
    add_subdirectory(hcc)
    target_link_libraries( kotekan_libs kotekan_hcc )
endif ()

if (${USE_HSA})
    add_subdirectory(hsa)
    link_directories ( /opt/rocm/hsa/lib/ )
    target_link_libraries( kotekan_libs kotekan_hsa )
    target_link_libraries( kotekan_libs hsa-runtime64 )
    include_directories ( /opt/rocm/hsa/include/ )
endif ()

if (${USE_OPENCL})
    add_subdirectory(opencl)
    target_link_libraries( kotekan_libs kotekan_opencl )
    target_link_libraries( kotekan_libs ${OpenCL_LIBRARIES} )
endif ()

if (${USE_DPDK})
    add_subdirectory(dpdk)
    target_link_libraries( kotekan_libs kotekan_dpdk )
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries( kotekan_libs -Wl,--no-whole-archive )
endif ()
### END SELF-REGISTERING BITS

# HDF5 stuff
if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  include_directories (/kotekan/HighFive/include/)
  target_link_libraries ( kotekan_libs ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
endif ()

add_subdirectory(version)
target_link_libraries( kotekan_libs kotekan_version )

