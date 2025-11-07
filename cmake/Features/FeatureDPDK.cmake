# Feature: DPDK
# Centralizes the tri-state handling for DPDK discovery and messaging.

# Inputs:
#  - USE_DPDK        : AUTO/ON/OFF toggle (AUTO probes via pkg-config)
#  - WITH_BOOST_TESTS: if ON, force DPDK OFF to avoid linker issues

# Outputs (for summary/consumers):
#  - DPDK_REASON : short human explanation for the status
#  - DPDK_FOUND : Whether DPDK was found (by find_package)

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

# If building Boost tests, always disable DPDK
if (WITH_BOOST_TESTS)
    set(USE_DPDK "OFF")
    set(DPDK_REASON "disabled (boost tests)")
    kmsg_warn("DPDK disabled while WITH_BOOST_TESTS=ON")
    return()
endif ()

macro(_kotekan_detect_dpdk OUT_FOUND)
    set(${OUT_FOUND} OFF)
    find_package(PkgConfig)
    if (NOT PKG_CONFIG_FOUND)
        return()
    endif ()

    pkg_check_modules(DPDK libdpdk>=19.11)
    if (DPDK_FOUND)
        set(${OUT_FOUND} ON)
    endif ()
endmacro()

if ("${USE_DPDK}" STREQUAL "AUTO")
    _kotekan_detect_dpdk(_dpdk_found)
    if (_dpdk_found)
        set(DPDK_REASON "auto-detected")
        kmsg_ok("DPDK found (autodetected): enabling DPDK components (disable with -DUSE_DPDK=OFF)")
    else ()
        if (PKG_CONFIG_FOUND)
            set(DPDK_REASON "disabled, not found")
            kmsg_warn("DPDK not found (default AUTO, continuing without). Requires libdpdk >= 19.11 via pkg-config.")
        else ()
            set(DPDK_REASON "disabled, pkg-config missing")
            kmsg_warn("pkg-config not available; cannot auto-detect DPDK. Install pkg-config or set -DUSE_DPDK=OFF.")
        endif ()
    endif ()
elseif ("${USE_DPDK}" STREQUAL "ON")
    _kotekan_detect_dpdk(_dpdk_found)
    if (_dpdk_found)
        set(DPDK_REASON "enabled, found")
        kmsg_ok("DPDK explicitly enabled via -DUSE_DPDK=ON")
    else ()
        if (PKG_CONFIG_FOUND)
            set(DPDK_REASON "enabled, not found")
            kmsg_error("DPDK not found when requested! Requires libdpdk >= 19.11 via pkg-config.")
        else ()
            set(DPDK_REASON "enabled, pkg-config missing")
            kmsg_error("pkg-config not available while DPDK requested; install pkg-config or disable DPDK.")
        endif ()
    endif ()
else ()
    set(DPDK_REASON "disabled")
    kmsg_status("DPDK explicitly disabled via -DUSE_DPDK=OFF")
endif ()

# NUMA is required for DPDK components; provide a clear error if toggled off
if (USE_DPDK AND DEFINED USE_NUMA AND (NOT USE_NUMA))
    kmsg_error("DPDK requires NUMA support. Enable it with -DUSE_NUMA=ON or disable DPDK via -DUSE_DPDK=OFF.")
endif ()
