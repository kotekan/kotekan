# There are no symbols in this entire library, and so it is created as an INTERFACE library. If any
# symbols are ever added it will need to be turned back into a standard CMake library
project(kotekan_libs)

set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

add_library(kotekan_libs INTERFACE)

# Julia is a covnenient language to implement stages for testing and development
find_package(Julia)
if(${Julia_FOUND})
    message("Julia support enabled")
    target_include_directories(kotekan_libs INTERFACE ${JULIA_INCLUDE_DIRS})
    target_link_libraries(kotekan_libs INTERFACE ${JULIA_LIBRARY})
    add_definitions(-DWITH_JULIA)
else()
    message("Julia not found, support disabled in kotekan_lib")
endif()

add_subdirectory(utils)
target_link_libraries(kotekan_libs INTERFACE kotekan_utils)

add_subdirectory(stages)
target_link_libraries(kotekan_libs INTERFACE kotekan_stages)

add_subdirectory(version)
target_link_libraries(kotekan_libs INTERFACE kotekan_version)

target_link_libraries(kotekan_libs INTERFACE libexternal)

if(${WITH_TESTS})
    add_subdirectory(testing)
    target_link_libraries(kotekan_libs INTERFACE kotekan_testing)
endif()

add_subdirectory(core)
target_link_libraries(kotekan_libs INTERFACE kotekan_core)

add_subdirectory(metadata)
target_link_libraries(kotekan_libs INTERFACE kotekan_metadata)

if(${USE_HSA}
   OR ${USE_OPENCL}
   OR ${USE_CUDA}
   OR ${USE_HIP})
    add_subdirectory(gpu)
endif()

if(${USE_HSA})
    add_subdirectory(hsa)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hsa kotekan_gpu)
endif()

if(${USE_OPENCL})
    add_subdirectory(opencl)
    target_link_libraries(kotekan_libs INTERFACE kotekan_opencl kotekan_gpu)
endif()

if(${USE_CUDA})
    add_subdirectory(cuda)
    target_link_libraries(kotekan_libs INTERFACE kotekan_cuda kotekan_gpu)
endif()

if(${USE_HIP})
    add_subdirectory(hip)
    target_link_libraries(kotekan_libs INTERFACE kotekan_hip kotekan_gpu)
endif()

# Include the old DPDK if the USE_OLD_DPDK flag is given.
if(${USE_OLD_DPDK})
    add_definitions(-DOLD_DPDK)
    add_subdirectory(dpdk)
    target_link_libraries(kotekan_libs INTERFACE kotekan_dpdk ${DPDK_LIBRARIES})
else()
    # Check for the new DPDK versions automatically
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(DPDK libdpdk>=19.11)
        if(DPDK_FOUND)
            message(STATUS "Found DPDK, including support for DPDK networking")
            add_subdirectory(dpdk)
            target_link_libraries(kotekan_libs INTERFACE kotekan_dpdk ${DPDK_LIBRARIES})
        else()
            message(
                STATUS "No DPDK package found, for older versions (<19.11) use -DUSE_OLD_DPDK=ON")
        endif()
    else()
        message(STATUS "pkg-config needed to include DPDK (>=19.11) support")
    endif()
endif()

# HDF5 stuff
if(${USE_HDF5})
    target_link_libraries(kotekan_libs INTERFACE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()
