# There are no symbols in this entire library, and so it is created as an INTERFACE library. If any
# symbols are ever added it will need to be turned back into a standard CMake library
project(kotekan_libs)

set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake ${KOTEKAN_SOURCE_DIR}/cmake/Modules)

# Julia is a convenient language to implement stages for testing and development
if(${USE_JULIA})
    find_package(Julia QUIET)
    if(Julia_FOUND)
        message(STATUS "Julia support enabled")
    else()
        message(WARNING "${ColorYellow}Julia not found; Julia features disabled in"
                        " kotekan_lib${ColorReset}")
    endif()
endif()

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

if(${USE_OPENCL}
   OR ${USE_CUDA}
   OR ${USE_HIP})
    add_subdirectory(gpu)
endif()

if(${USE_OPENCL})
    add_subdirectory(opencl)
    target_link_libraries(kotekan_libs INTERFACE kotekan_opencl)
endif()

if(${USE_CUDA})
    add_subdirectory(cuda)
    target_link_libraries(kotekan_libs INTERFACE kotekan_cuda)
endif()

if(${USE_HIP})
    add_subdirectory(hip)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hip)
endif()

if(${USE_OPENCL}
   OR ${USE_CUDA}
   OR ${USE_HIP})
    target_link_libraries(kotekan_libs INTERFACE kotekan_gpu)
endif()

# DPDK selection and summary handled in FeatureDPDK.cmake. Here we honor the effective decision and
# wire targets accordingly.
if(DEFINED DPDK_ENABLED AND DPDK_ENABLED)
    if(DEFINED KOTEKAN_DPDK_MODE AND (KOTEKAN_DPDK_MODE STREQUAL "OLD"))
        add_definitions(-DOLD_DPDK)
    endif()
    add_subdirectory(dpdk)
    target_link_libraries(kotekan_libs INTERFACE kotekan_dpdk ${DPDK_LIBRARIES})
endif()

# HDF5 stuff
if(${USE_HDF5})
    target_link_libraries(kotekan_libs INTERFACE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()
