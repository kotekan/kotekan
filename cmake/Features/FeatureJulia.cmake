# Julia feature detection and setup

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

set(JULIA_ENABLED OFF)
set(JULIA_REASON "disabled")
if(${USE_JULIA})
    # Best practice: use the provided FindJulia.cmake when the Julia executable can run
    find_program(_JULIA_EXE julia)
    set(_Julia_can_run OFF)
    if(_JULIA_EXE)
        execute_process(COMMAND "${_JULIA_EXE}" --startup-file=no --version
                        RESULT_VARIABLE _JULIA_RV OUTPUT_QUIET ERROR_QUIET)
        if(_JULIA_RV EQUAL 0)
            set(_Julia_can_run ON)
        endif()
    endif()

    if(_Julia_can_run)
        find_package(Julia QUIET)
    else()
        set(Julia_FOUND OFF)
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
    set(Julia_FOUND OFF)
    kmsg_status("Julia disabled via -DUSE_JULIA=OFF")
endif()
