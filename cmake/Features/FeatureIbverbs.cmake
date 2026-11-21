# Feature: ibverbs
# Centralizes the tri-state handling for libibverbs discovery and messaging. libibverbs backs the
# raw packet UDP stages in lib/ibverbs.

# Inputs:
#  - USE_IBVERBS : AUTO/ON/OFF toggle (AUTO probes via pkg-config)

# Outputs (for summary/consumers):
#  - IBVERBS_REASON : short human explanation for the status
#  - IBVERBS_FOUND : Whether libibverbs was found (by pkg-config)

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

macro(_kotekan_detect_ibverbs OUT_FOUND)
    set(${OUT_FOUND} OFF)
    find_package(PkgConfig)
    if(NOT PKG_CONFIG_FOUND)
        return()
    endif()

    pkg_check_modules(IBVERBS libibverbs)
    if(IBVERBS_FOUND)
        set(${OUT_FOUND} ON)
    endif()
endmacro()

if("${USE_IBVERBS}" STREQUAL "AUTO")
    _kotekan_detect_ibverbs(_ibverbs_found)
    if(_ibverbs_found)
        set(IBVERBS_REASON "auto-detected")
        kmsg_ok("libibverbs found (autodetected): enabling ibverbs stages (disable with -DUSE_IBVERBS=OFF)")
    else()
        if(PKG_CONFIG_FOUND)
            set(IBVERBS_REASON "disabled, not found")
            kmsg_warn("libibverbs not found (default AUTO, continuing without). Requires libibverbs via pkg-config (package libibverbs-dev).")
        else()
            set(IBVERBS_REASON "disabled, pkg-config missing")
            kmsg_warn("pkg-config not available; cannot auto-detect libibverbs. Install pkg-config or set -DUSE_IBVERBS=OFF.")
        endif()
    endif()
elseif("${USE_IBVERBS}" STREQUAL "ON")
    _kotekan_detect_ibverbs(_ibverbs_found)
    if(_ibverbs_found)
        set(IBVERBS_REASON "enabled, found")
        kmsg_ok("ibverbs explicitly enabled via -DUSE_IBVERBS=ON")
    else()
        set(IBVERBS_REASON "enabled, not found")
        kmsg_error("libibverbs not found when requested! Requires libibverbs via pkg-config (package libibverbs-dev).")
    endif()
else()
    set(IBVERBS_REASON "disabled")
    kmsg_status("ibverbs explicitly disabled via -DUSE_IBVERBS=OFF")
endif()
