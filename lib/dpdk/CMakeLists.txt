project(kotekan_libs)

set(KOTEKAN_DPDK_SOURCES
        dpdkCore.cpp
        zeroSamples.cpp
        invalidateVDIFframes.cpp
        )

add_library(kotekan_dpdk ${KOTEKAN_DPDK_SOURCES})

# Link in DPDK
find_library(dpdk REQUIRED)
message("DPDK include dir: ${DPDK_INCLUDE_DIR}")
include_directories(SYSTEM ${DPDK_INCLUDE_DIR})
target_link_libraries(kotekan_dpdk ${DPDK_LIBRARIES})

target_link_libraries(kotekan_dpdk libexternal)

add_definitions(-DWITH_DPDK)

if(NOT ${NUMA_FOUND})
    message( FATAL_ERROR "ERROR: libnuma is required for DPDK.  Please install libnuma")
endif()
