project(kotekan_libs)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

# Link in DPDK
find_library(dpdk REQUIRED)
message("DPDK include dir: ${DPDK_INCLUDE_DIR}")
target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIR})
target_link_libraries(kotekan_dpdk PRIVATE ${DPDK_LIBRARIES} ${NUMA_LIBRARY} libexternal
                                           kotekan_libs)
target_include_directories(kotekan_dpdk PUBLIC .)

add_definitions(-DWITH_DPDK)

find_package(NUMA REQUIRED)
