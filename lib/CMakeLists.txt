# There are no symbols in this entire library, and so it is created as an INTERFACE library. If any
# symbols are ever added it will need to be turned back into a standard CMake library
project(kotekan_libs)

set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake ${KOTEKAN_SOURCE_DIR}/cmake/Modules)

add_library(kotekan_libs INTERFACE)

add_subdirectory(utils)
target_link_libraries(kotekan_libs INTERFACE kotekan_utils)

add_subdirectory(core)
target_link_libraries(kotekan_libs INTERFACE kotekan_core)

add_subdirectory(stages)
target_link_libraries(kotekan_libs INTERFACE kotekan_stages)

add_subdirectory(version)
target_link_libraries(kotekan_libs INTERFACE kotekan_version)

target_link_libraries(kotekan_libs INTERFACE libexternal)

# Build C++ testing helpers when requested. WITH_TESTS builds and links the lib/testing helper
# stages into the main kotekan binary. WITH_BOOST_TESTS also requires these helpers and additionally
# builds the unit tests under tests/boost.
if(${WITH_TESTS} OR ${WITH_BOOST_TESTS})
    add_subdirectory(testing)
    target_link_libraries(kotekan_libs INTERFACE kotekan_testing)
endif()

add_subdirectory(metadata)
target_link_libraries(kotekan_libs INTERFACE kotekan_metadata)

if(OPENCL_FOUND
   OR CUDAToolkit_FOUND
   OR HIP_FOUND)
    add_subdirectory(gpu)
endif()

if(OPENCL_FOUND)
    add_subdirectory(opencl)
    target_link_libraries(kotekan_libs INTERFACE kotekan_opencl)
endif()

if(CUDAToolkit_FOUND)
    add_subdirectory(cuda)
    target_link_libraries(kotekan_libs INTERFACE kotekan_cuda)
endif()

if(HIP_FOUND)
    add_subdirectory(hip)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hip)
endif()

if(OPENCL_FOUND
   OR CUDAToolkit_FOUND
   OR HIP_FOUND)
    target_link_libraries(kotekan_libs INTERFACE kotekan_gpu)
endif()

# DPDK selection is handled in FeatureDPDK.cmake. Only include/link when DPDK was found and the
# toggle is AUTO or ON.
if(DPDK_FOUND AND ("${USE_DPDK}" STREQUAL "AUTO" OR "${USE_DPDK}" STREQUAL "ON"))
    add_subdirectory(dpdk)
    target_link_libraries(kotekan_libs INTERFACE kotekan_dpdk ${DPDK_LIBRARIES})
endif()

# HDF5 stuff
if(HDF5_FOUND
   AND HighFive_FOUND
   AND HDF5Plugin_PLUGIN_DIR)
    target_link_libraries(kotekan_libs INTERFACE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()
