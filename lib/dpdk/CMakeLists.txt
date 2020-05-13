project(kotekan_libs)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

# Link in DPDK
find_library(dpdk REQUIRED)
message("DPDK include dir: ${DPDK_INCLUDE_DIR}")
include_directories(SYSTEM ${DPDK_INCLUDE_DIR})
target_link_libraries(kotekan_dpdk ${DPDK_LIBRARIES})

target_link_libraries(kotekan_dpdk libexternal)

add_definitions(-DWITH_DPDK)

set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(NUMA REQUIRED)
link_libraries(${NUMA_LIBRARY})
