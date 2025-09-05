## Feature: DPDK
# Centralize DPDK enable/disable and version selection logic.

# Inputs:
#  - USE_DPDK: STRING one of ON (auto, default), OFF, NEW, OLD
#  - WITH_BOOST_TESTS: if ON, force DPDK off (to avoid linker issues)
#  - USE_OLD_DPDK: legacy boolean flag (maps to USE_DPDK=OLD if set)

# Outputs (cached summary variables used by Summary.cmake):
#  - DPDK_ENABLED: BOOL whether we plan to build with DPDK
#  - DPDK_REASON: STRING short human explanation
#  - KOTEKAN_DPDK_MODE: STRING one of OFF, NEW, OLD (effective mode)

include_guard(GLOBAL)

# Normalize option values
set(_kotekan_use_dpdk "${USE_DPDK}")
string(TOUPPER "${_kotekan_use_dpdk}" _kotekan_use_dpdk)
if(NOT _kotekan_use_dpdk)
    set(_kotekan_use_dpdk "ON")
endif()

# Backward-compat: legacy USE_OLD_DPDK=ON implies OLD, unless user explicitly requested something
if(DEFINED USE_OLD_DPDK AND USE_OLD_DPDK AND ("${_kotekan_use_dpdk}" STREQUAL "ON"))
    set(_kotekan_use_dpdk "OLD")
    message(WARNING "USE_OLD_DPDK is deprecated; use -DUSE_DPDK=OLD instead")
endif()

# If building Boost tests, always disable DPDK
if(WITH_BOOST_TESTS)
    set(DPDK_ENABLED OFF CACHE BOOL "DPDK available" FORCE)
    set(DPDK_REASON "disabled (boost tests)" CACHE STRING "" FORCE)
    set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
    set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW OLD)
    return()
endif()

# Default summary outputs
set(DPDK_ENABLED OFF CACHE BOOL "DPDK available")
set(DPDK_REASON "disabled" CACHE STRING "" FORCE)
set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW OLD)

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
            set_property(CACHE KOTEKAN_DPDK_MODE PROPERTY STRINGS OFF NEW OLD)
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
elseif("${_kotekan_use_dpdk}" STREQUAL "OLD")
    # Force OLD; actual discovery occurs in lib/dpdk/CMakeLists (FindDPDK.cmake)
    # Keep DPDK_FOUND unset so subdir uses FindDPDK path
    set(DPDK_ENABLED ON CACHE BOOL "DPDK available" FORCE)
    set(DPDK_REASON "old DPDK selected (-DUSE_DPDK=OLD)" CACHE STRING "" FORCE)
    set(KOTEKAN_DPDK_MODE OLD CACHE STRING "Effective DPDK mode" FORCE)
else()
    # ON/AUTO: prefer NEW via pkg-config; if not available, try to detect OLD, else disable
    _kotekan_try_dpdk_new()
    if(DPDK_ENABLED)
        # Already configured as NEW
    else()
        # Probe legacy installs via our FindDPDK module. Only enable if fully found.
        # This may populate DPDK_* variables; downstream logic will use KOTEKAN_DPDK_MODE.
        find_package(DPDK QUIET)
        if(DPDK_FOUND)
            set(DPDK_ENABLED ON CACHE BOOL "DPDK available" FORCE)
            set(DPDK_REASON "auto: using old DPDK (FindDPDK)" CACHE STRING "" FORCE)
            set(KOTEKAN_DPDK_MODE OLD CACHE STRING "Effective DPDK mode" FORCE)
        else()
            set(DPDK_ENABLED OFF CACHE BOOL "DPDK available" FORCE)
            set(DPDK_REASON "not found (NEW via pkg-config nor OLD via FindDPDK)" CACHE STRING "" FORCE)
            set(KOTEKAN_DPDK_MODE OFF CACHE STRING "Effective DPDK mode" FORCE)
        endif()
    endif()
endif()

# Persist normalized request for downstream logic
set(USE_DPDK "${_kotekan_use_dpdk}" CACHE STRING "DPDK usage: ON (auto), OFF, NEW, OLD")
set_property(CACHE USE_DPDK PROPERTY STRINGS ON OFF NEW OLD)

# NUMA is required for DPDK components; provide a clear error if toggled off
if(DPDK_ENABLED AND DEFINED USE_NUMA AND (NOT USE_NUMA))
    message(FATAL_ERROR "DPDK requires NUMA support. Enable it with -DUSE_NUMA=ON or disable DPDK via -DUSE_DPDK=OFF.")
endif()
