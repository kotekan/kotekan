project(kotekan_dpdk)

find_package(NUMA REQUIRED)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

# Honour the decision made in FeatureDPDK.cmake and ensure pkg-config data is available.
if(NOT (DEFINED DPDK_ENABLED AND DPDK_ENABLED))
    message(FATAL_ERROR "kotekan_dpdk requested but DPDK support is disabled")
endif()

if(NOT DPDK_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK REQUIRED libdpdk>=19.11)
endif()

add_definitions(${DPDK_STATIC_CFLAGS})
set(DPDK_LIBRARIES ${DPDK_STATIC_LDFLAGS})
message(STATUS "DPDK include dirs: ${DPDK_INCLUDE_DIRS}")
target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIRS})

target_link_libraries(kotekan_dpdk PRIVATE ${DPDK_LIBRARIES} ${NUMA_LIBRARY} libexternal
                                           kotekan_libs)
target_include_directories(kotekan_dpdk PUBLIC .)
