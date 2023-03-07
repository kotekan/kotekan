project(kotekan_dpdk)

find_package(NUMA REQUIRED)

add_library(kotekan_dpdk dpdkCore.cpp zeroSamples.cpp invalidateVDIFframes.cpp)

find_package(PkgConfig REQUIRED)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(DPDK "libdpdk")
    if (DPDK_FOUND)
        message(STATUS "found dpdk via pkg-config")
        add_definitions(${DPDK_STATIC_CFLAGS})
        set(DPDK_LIBRARIES ${DPDK_STATIC_LDFLAGS})
        add_definitions(-DHAVE_DPDK)
    endif ()
endif ()

# Link in DPDK
find_package(DPDK REQUIRED)
message("DPDK include dir: ${DPDK_INCLUDE_DIR}")
target_include_directories(kotekan_dpdk SYSTEM PRIVATE ${DPDK_INCLUDE_DIR})
target_link_libraries(kotekan_dpdk PRIVATE ${DPDK_LIBRARIES} ${NUMA_LIBRARY} libexternal
        kotekan_libs)
target_include_directories(kotekan_dpdk PUBLIC .)
