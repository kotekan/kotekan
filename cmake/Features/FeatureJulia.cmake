# Julia feature detection.
# Resolves the USE_JULIA tri-state option via executable and package checks,

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

set(JULIA_REASON "disabled")

macro(_kotekan_detect_julia OUT_FOUND OUT_REASON)
    set(${OUT_FOUND} OFF)
    set(${OUT_REASON} "executable not found")

    find_program(_julia_exe julia)
    if (NOT _julia_exe)
        return()
    endif ()

    execute_process(
            COMMAND "${_julia_exe}" --startup-file=no --version
            RESULT_VARIABLE _julia_rv
            OUTPUT_QUIET
            ERROR_QUIET)
    if (NOT _julia_rv EQUAL 0)
        set(${OUT_REASON} "executable not runnable")
        return()
    endif ()

    find_package(Julia)
    if (Julia_FOUND)
        set(${OUT_FOUND} ON)
        set(${OUT_REASON} "found")
    else ()
        set(${OUT_REASON} "not found")
    endif ()
endmacro()

if ("${USE_JULIA}" STREQUAL "AUTO")
    _kotekan_detect_julia(_julia_found _julia_reason)
    if (_julia_found)
        set(USE_JULIA ON CACHE BOOL "Use Julia features (autodetected)" FORCE)
        set(JULIA_REASON "auto-detected")
        kmsg_ok("Julia found (autodetected): enabling Julia features (disable with -DUSE_JULIA=OFF)")
    else ()
        set(USE_JULIA OFF CACHE BOOL "Use Julia features (autodetected)" FORCE)
        set(JULIA_REASON "disabled, ${_julia_reason}")
        kmsg_warn("Julia ${_julia_reason} (default AUTO, continuing without).")
    endif ()
elseif ("${USE_JULIA}" STREQUAL "ON")
    _kotekan_detect_julia(_julia_found _julia_reason)
    if (_julia_found)
        set(JULIA_REASON "enabled, found")
        kmsg_ok("Julia explicitly enabled via -DUSE_JULIA=ON")
    else ()
        set(JULIA_REASON "enabled, ${_julia_reason}")
        kmsg_error("Julia ${_julia_reason} when requested! Disable with -DUSE_JULIA=OFF.")
    endif ()
else ()
    set(JULIA_REASON "disabled")
    kmsg_status("Julia explicitly disabled via -DUSE_JULIA=OFF")
endif ()
