# Math feature detection for FFTW and LAPACKE/Blaze.
# Converts the USE_* tri-state toggles into ON/OFF decisions, captures reason
# strings for the summary, and mirrors legacy *_ENABLED cache variables.

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/../Color.cmake)

# FFTW
if("${USE_FFTW}" STREQUAL "AUTO" OR "${USE_FFTW}" STREQUAL "ON")
    find_package(FFTW)
    if(FFTW_FOUND)
        add_definitions(-DWITH_FFTW)
        if("${USE_FFTW}" STREQUAL "AUTO")
            set(FFTW_REASON "auto-detected")
            kmsg_ok("FFTW found (autodetected): enabling FFTW F-engine (disable with -DUSE_FFTW=OFF)")
        else()
            set(FFTW_REASON "enabled, found")
            kmsg_ok("FFTW explicitly enabled via -DUSE_FFTW=ON")
        endif()
    else()
        if("${USE_FFTW}" STREQUAL "AUTO")
            set(FFTW_REASON "disabled, not found")
            kmsg_warn("FFTW not found (default AUTO, continuing without).")
        else()
            set(FFTW_REASON "enabled, not found")
            kmsg_error("FFTW not found when requested! Disable with -DUSE_FFTW=OFF.")
        endif()
    endif()
else()
    set(FFTW_REASON "disabled")
    kmsg_status("FFTW explicitly disabled via -DUSE_FFTW=OFF")
endif()


# LAPACKE + Blaze
if("${USE_LAPACK_BLAZE}" STREQUAL "AUTO" OR "${USE_LAPACK_BLAZE}" STREQUAL "ON")
    find_package(LAPACKE)
    find_package(Blaze)

    set(_missing_lapack_blaze "")
    if(NOT LAPACKE_FOUND)
        list(APPEND _missing_lapack_blaze "LAPACKE")
    endif()
    if(NOT BLAZE_FOUND)
        list(APPEND _missing_lapack_blaze "Blaze headers")
    endif()

    if(_missing_lapack_blaze STREQUAL "")
        add_definitions(-DBLAZE_BLAS_MODE=1)
        add_definitions(-DBLAZE_BLAS_IS_PARALLEL=1)

        if("${USE_LAPACK_BLAZE}" STREQUAL "AUTO")
            set(LAPACK_BLAZE_REASON "auto-detected")
            kmsg_ok(
                "LAPACK/Blaze found (autodetected): enabling linear algebra stages (disable with -DUSE_LAPACK_BLAZE=OFF)")
        else()
            set(LAPACK_BLAZE_REASON "enabled, found")
            kmsg_ok("LAPACK/Blaze explicitly enabled via -DUSE_LAPACK_BLAZE=ON")
        endif()

        kmsg_status("Using LAPACKE includes ${LAPACKE_INCLUDE_DIRS}")
        kmsg_status("Using LAPACKE libraries ${LAPACKE_LIBRARIES}")
        kmsg_status("Blaze found. BLAZE_PATH is ${BLAZE_PATH}")

        # Ensure OpenBLAS is linked if generic BLAS was selected by LAPACKE but
        # code uses OpenBLAS API
        string(JOIN ";" _lapacke_libs_str ${LAPACKE_LIBRARIES})
        if(NOT _lapacke_libs_str MATCHES "openblas")
            find_library(
                OPENBLAS_EXTRA_LIB
                NAMES openblas
                PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
            if(OPENBLAS_EXTRA_LIB)
                set(OPENBLAS_EXTRA_LIB ${OPENBLAS_EXTRA_LIB}
                    CACHE FILEPATH "OpenBLAS lib for explicit linking")
                kmsg_status(
                    "Detected generic BLAS; adding OpenBLAS explicitly: ${OPENBLAS_EXTRA_LIB}")
            endif()
        endif()
    else()
        if("${USE_LAPACK_BLAZE}" STREQUAL "AUTO")
            set(LAPACK_BLAZE_REASON "disabled, missing: ${_missing_lapack_blaze}")
            kmsg_warn(
                "LAPACK/Blaze not fully found (missing: ${_missing_lapack_blaze}); LAPACK/Blaze stages disabled.")
        else()
            set(LAPACK_BLAZE_REASON "enabled, missing: ${_missing_lapack_blaze}")
            kmsg_error(
                "LAPACK/Blaze not fully found (missing: ${_missing_lapack_blaze}) when requested! Disable with -DUSE_LAPACK_BLAZE=OFF.")
        endif()
        kmsg_status("To skip these checks, configure with -DUSE_LAPACK_BLAZE=OFF")
    endif()
else()
    set(LAPACK_BLAZE_REASON "disabled")
    kmsg_status("LAPACK/Blaze explicitly disabled via -DUSE_LAPACK_BLAZE=OFF")
endif()

# Mirror legacy variable for versioning/tests that expect USE_LAPACK
set(USE_LAPACK ${USE_LAPACK_BLAZE} CACHE STRING "Mirror of USE_LAPACK_BLAZE (compat)" FORCE)
