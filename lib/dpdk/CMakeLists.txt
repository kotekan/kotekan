project(kotekan_dpdk)

find_package(NUMA REQUIRED)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

# Select integration path based on effective mode from FeatureDPDK.cmake
if(DEFINED KOTEKAN_DPDK_MODE AND (KOTEKAN_DPDK_MODE STREQUAL "NEW"))
    # NEW: use pkg-config provided flags from FeatureDPDK. If missing (rare), require them here.
    if(NOT DPDK_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(DPDK REQUIRED libdpdk>=19.11)
    endif()
    add_definitions(${DPDK_STATIC_CFLAGS})
    set(DPDK_LIBRARIES ${DPDK_STATIC_LDFLAGS})
    message(STATUS "DPDK include dirs: ${DPDK_INCLUDE_DIRS}")
    target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIRS})
elseif(DEFINED KOTEKAN_DPDK_MODE AND (KOTEKAN_DPDK_MODE STREQUAL "OLD"))
    # OLD: rely on our FindDPDK module
    find_package(DPDK REQUIRED)
    message(STATUS "DPDK include dir: ${DPDK_INCLUDE_DIR}")
    target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIR})
else()
    message(FATAL_ERROR "kotekan_dpdk requested but KOTEKAN_DPDK_MODE is not NEW or OLD")
endif()

target_link_libraries(kotekan_dpdk PRIVATE ${DPDK_LIBRARIES} ${NUMA_LIBRARY} libexternal kotekan_libs)
target_include_directories(kotekan_dpdk PUBLIC .)
