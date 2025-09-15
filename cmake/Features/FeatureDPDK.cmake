## Feature: DPDK
# Centralize DPDK enable/disable and version selection logic.

# Inputs:
#  - USE_DPDK: STRING one of ON (auto, default) or OFF
#  - WITH_BOOST_TESTS: if ON, force DPDK off (to avoid linker issues)

# Outputs (cached summary variables used by Summary.cmake):
#  - DPDK_ENABLED: BOOL whether we plan to build with DPDK
#  - DPDK_REASON: STRING short human explanation
#  - KOTEKAN_DPDK_MODE: STRING OFF or NEW (effective mode)

include_guard(GLOBAL)

# Normalize option values
set(_kotekan_use_dpdk "${USE_DPDK}")
string(TOUPPER "${_kotekan_use_dpdk}" _kotekan_use_dpdk)
if(NOT _kotekan_use_dpdk)
    set(_kotekan_use_dpdk "ON")
endif()

# If building Boost tests, always disable DPDK
if(WITH_BOOST_TESTS)
    set(DPDK_ENABLED OFF CACHE BOOL "DPDK available" FORCE)
    set(DPDK_REASON "disabled (boost tests)" CACHE STRING "" FORCE)
    set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
    set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW)
    return()
endif()

# Default summary outputs
set(DPDK_ENABLED OFF CACHE BOOL "DPDK available")
set(DPDK_REASON "disabled" CACHE STRING "" FORCE)
set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW)

# Helper: try pkg-config for NEW DPDK (>=19.11)
macro(_kotekan_try_dpdk_new)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        # Prefer consolidated libdpdk (>=19.11) via pkg-config
        pkg_check_modules(DPDK libdpdk>=19.11)
        if(DPDK_FOUND)
            set(DPDK_ENABLED ON CACHE BOOL "DPDK available" FORCE)
            set(DPDK_REASON "found (>=19.11) via pkg-config" CACHE STRING "" FORCE)
            set(KOTEKAN_DPDK_MODE NEW CACHE STRING "Effective DPDK mode" FORCE)
            set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW)
        endif()
    endif()
endmacro()

# Resolve requested mode
if("${_kotekan_use_dpdk}" STREQUAL "OFF")
    set(DPDK_ENABLED OFF CACHE BOOL "DPDK available" FORCE)
    set(DPDK_REASON "disabled (-DUSE_DPDK=OFF)" CACHE STRING "" FORCE)
    set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
elseif("${_kotekan_use_dpdk}" STREQUAL "NEW")
    # Require NEW; if not found leave disabled with message
    _kotekan_try_dpdk_new()
    if(NOT DPDK_ENABLED)
        set(DPDK_REASON "requested NEW not found (>=19.11)" CACHE STRING "" FORCE)
        set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
    endif()
else()
    # ON/AUTO: prefer NEW via pkg-config; if not available, disable
    _kotekan_try_dpdk_new()
    if(DPDK_ENABLED)
        # Already configured as NEW
    else()
        set(DPDK_ENABLED OFF CACHE BOOL "DPDK available" FORCE)
        set(DPDK_REASON "not found (requires libdpdk >= 19.11 via pkg-config)"
            CACHE STRING "" FORCE)
        set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
    endif()
endif()

# Persist normalized request for downstream logic
set(USE_DPDK "${_kotekan_use_dpdk}" CACHE STRING "DPDK usage: ON (auto), OFF")
set_property(CACHE USE_DPDK PROPERTY STRINGS ON OFF)

# NUMA is required for DPDK components; provide a clear error if toggled off
if(DPDK_ENABLED AND DEFINED USE_NUMA AND (NOT USE_NUMA))
    message(FATAL_ERROR "DPDK requires NUMA support. Enable it with -DUSE_NUMA=ON or disable DPDK via -DUSE_DPDK=OFF.")
endif()
