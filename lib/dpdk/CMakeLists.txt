project(kotekan_dpdk)

# Assumes FeatureDPDK.cmake already ran at the top level and populated DPDK_* variables via
# pkg-config when enabled and detected.

find_package(NUMA REQUIRED)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

add_definitions(${DPDK_STATIC_CFLAGS})
set(DPDK_LIBRARIES ${DPDK_STATIC_LDFLAGS})
message(STATUS "DPDK include dirs: ${DPDK_INCLUDE_DIRS}")
target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIRS})

# Expose DPDK and NUMA to dependents to avoid duplicate link flag injection.
target_link_libraries(
    kotekan_dpdk
    PUBLIC ${DPDK_LIBRARIES} ${NUMA_LIBRARY}
    PRIVATE libexternal kotekan_libs)
target_include_directories(kotekan_dpdk PUBLIC .)
