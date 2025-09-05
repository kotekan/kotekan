# Julia feature detection and setup

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

set(JULIA_ENABLED OFF)
set(JULIA_REASON "disabled")
if(${USE_JULIA})
    # Only try FindJulia if the julia executable can run
    find_program(_julia_exe julia)
    set(_julia_can_run OFF)
    if(_julia_exe)
        execute_process(COMMAND "${_julia_exe}" --startup-file=no --version
                        RESULT_VARIABLE _julia_rv OUTPUT_QUIET ERROR_QUIET)
        if(_julia_rv EQUAL 0)
            set(_julia_can_run ON)
        endif()
    endif()

    if(_julia_can_run)
        find_package(Julia QUIET)
    endif()

    if(Julia_FOUND)
        set(JULIA_ENABLED ON)
        set(JULIA_REASON "found")
        kmsg_ok("Julia found: enabling Julia features (disable with -DUSE_JULIA=OFF)")
    else()
        set(USE_JULIA OFF)
        set(JULIA_ENABLED OFF)
        set(JULIA_REASON "not found/not runnable")
        # No immediate warning; the final summary will report this once.
    endif()
else()
    kmsg_status("Julia disabled via -DUSE_JULIA=OFF")
endif()
