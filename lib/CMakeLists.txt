# There are no symbols in this entire library, and so it is created as an INTERFACE library. If any
# symbols are ever added it will need to be turned back into a standard CMake library
project(kotekan_libs)

include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/metadata)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/stages)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/dpdk)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/cuda)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories(${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

add_library(kotekan_libs INTERFACE)

add_subdirectory(core)
target_link_libraries(kotekan_libs INTERFACE kotekan_core)

add_subdirectory(metadata)
target_link_libraries(kotekan_libs INTERFACE kotekan_metadata)

add_subdirectory(utils)
target_link_libraries(kotekan_libs INTERFACE kotekan_utils)

# SELF-REGISTERING BITS:
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(kotekan_libs INTERFACE -Wl,--whole-archive)
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(kotekan_libs INTERFACE -Wl,-all_load)
endif()

add_subdirectory(testing)
target_link_libraries(kotekan_libs INTERFACE kotekan_testing)

add_subdirectory(stages)
target_link_libraries(kotekan_libs INTERFACE kotekan_stages)

if(${USE_HCC})
    add_subdirectory(hcc)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hcc)
endif()

if(${USE_HSA}
   OR ${USE_OPENCL}
   OR ${USE_CUDA})
    add_subdirectory(gpu)
endif()

if(${USE_HSA})
    add_subdirectory(hsa)
    link_directories(/opt/rocm/hsa/lib/)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hsa kotekan_gpu)
    target_link_libraries(kotekan_libs INTERFACE hsa-runtime64)
    include_directories(/opt/rocm/hsa/include/)
endif()

if(${USE_OPENCL})
    add_subdirectory(opencl)
    target_link_libraries(kotekan_libs INTERFACE kotekan_opencl kotekan_gpu)
    target_link_libraries(kotekan_libs INTERFACE ${OpenCL_LIBRARY})
endif()

if(${USE_CUDA})
    add_subdirectory(cuda)
    target_link_libraries(kotekan_libs INTERFACE kotekan_cuda kotekan_gpu)
    target_link_libraries(kotekan_libs INTERFACE ${CUDA_LIBRARIES})
endif()

if(${USE_DPDK})
    add_subdirectory(dpdk)
    target_link_libraries(kotekan_libs INTERFACE kotekan_dpdk)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(kotekan_libs INTERFACE -Wl,--no-whole-archive)
endif()
# END SELF-REGISTERING BITS

# HDF5 stuff
if(${USE_HDF5})
    include_directories(${HDF5_INCLUDE_DIRS})
    include_directories(SYSTEM /kotekan/HighFive/include/)
    target_link_libraries(kotekan_libs INTERFACE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()

add_subdirectory(version)

target_link_libraries(kotekan_libs INTERFACE kotekan_version)
target_link_libraries(kotekan_libs INTERFACE libexternal)
