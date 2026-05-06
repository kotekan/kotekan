# All user-facing build options (no logic here).
#
# Developer note: when adding a new option, also update the configure summary (cmake/Summary.cmake),
# the documentation (Sphinx CMake option pages), and the version metadata
# (lib/version/version.c.in).

# ktk_tristate_option(<FLAG> <DESC> <DEFAULT> [EXPORTS] [PREFIX <name>])
#
# * <FLAG>    : cache variable name (e.g., USE_ASDF)
# * <DESC>    : help text
# * <DEFAULT> : one of AUTO, ON, OFF  (case-insensitive)
#
# Optional: EXPORTS         -> initializes <PREFIX>_ENABLED and <PREFIX>_REASON in PARENT_SCOPE
# PREFIX <name>   -> export prefix (defaults to <FLAG>)
#
# Notes: - Does not overwrite an existing cache value (no FORCE). - Sets allowed values list in
# cmake-gui/ccmake.

# ktk_tristate_option(<FLAG> <DESC> <DEFAULT> [EXPORTS] [PREFIX <name>])
#
# * <DEFAULT>: one of AUTO, ON, OFF (case-insensitive; boolean synonyms OK)
# * Accepts user -D<FLAG>=<value> with common CMake boolean synonyms: ON:  ON TRUE YES Y 1 OFF: OFF
#   FALSE NO  N 0 AUTO: AUTO
# * Canonicalizes the cache entry to one of: AUTO / ON / OFF
#
function(ktk_tristate_option flag desc default)
    set(options EXPORTS)
    set(oneValueArgs PREFIX)
    cmake_parse_arguments(KTK "${options}" "${oneValueArgs}" "" ${ARGN})

    # Normalization helper
    function(_ktk_norm in outvar)
        string(STRIP "${in}" s_)
        string(TOUPPER "${s_}" u_)
        set(on_ "ON;TRUE;YES;Y;1")
        set(off_ "OFF;FALSE;NO;N;0")
        if(u_ STREQUAL "AUTO")
            set(${outvar}
                "AUTO"
                PARENT_SCOPE)
        elseif(u_ IN_LIST on_)
            set(${outvar}
                "ON"
                PARENT_SCOPE)
        elseif(u_ IN_LIST off_)
            set(${outvar}
                "OFF"
                PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid value '${in}' for ${flag}. Use one of: AUTO, ON, OFF "
                                "(ON aliases: ON/TRUE/YES/Y/1; OFF aliases: OFF/FALSE/NO/N/0).")
        endif()
    endfunction()

    # Normalize default and ensure cache entry exists
    _ktk_norm("${default}" _def)
    if(NOT DEFINED CACHE{${flag}})
        set(${flag}
            "${_def}"
            CACHE STRING "${desc} (AUTO/ON/OFF)")
    endif()

    # Validate/normalize existing user-provided value, then canonicalize in cache
    get_property(
        _cur
        CACHE ${flag}
        PROPERTY VALUE)
    _ktk_norm("${_cur}" _canon)
    if(NOT _canon STREQUAL "${_cur}")
        # Canonicalize to AUTO/ON/OFF in cache so GUIs and scripts see the clean form
        set(${flag}
            "${_canon}"
            CACHE STRING "${desc} (AUTO/ON/OFF)" FORCE)
    endif()

    # Present choices in GUIs
    set_property(CACHE ${flag} PROPERTY STRINGS AUTO ON OFF)
endfunction()

# AUTO by default (most "Features")
ktk_tristate_option(USE_AIRSPY "Build Airspy Producer" AUTO)
ktk_tristate_option(USE_ASDF "Build ASDF output stages" AUTO)
ktk_tristate_option(USE_CUDA "Build CUDA GPU Framework" AUTO)
ktk_tristate_option(USE_DPDK "Build with DPDK libraries" AUTO)
ktk_tristate_option(USE_FFTW "Build with FFTW" AUTO)
ktk_tristate_option(USE_GDAL "Build GDAL output stages" AUTO)
ktk_tristate_option(USE_HDF5 "Build HDF5 output stages" AUTO)
ktk_tristate_option(USE_JULIA "Build Julia-based features" AUTO)
ktk_tristate_option(USE_LAPACK_BLAZE
                    "Build with LAPACK Linear Algebra (OpenBLAS) and Blaze support" AUTO)
ktk_tristate_option(USE_OMP "Enable OpenMP" AUTO)
ktk_tristate_option(USE_OPENSSL "Enable OpenSSL (hash) support in core" AUTO)

# ON by default
ktk_tristate_option(USE_NUMA "Enable NUMA support in core (libnuma)" ON)
ktk_tristate_option(WERROR "Warnings are errors" ON)

# OFF by default
ktk_tristate_option(USE_LTO "Enable link-time optimization in Release builds" OFF)
ktk_tristate_option(CCACHE "Use ccache to speed up the build" OFF)
ktk_tristate_option(CMAKE_LINK_WHAT_YOU_USE "Report missing link dependencies while building" OFF)
ktk_tristate_option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)
ktk_tristate_option(IWYU "Enable include-what-you-use and print suggestions to stderr" OFF)
ktk_tristate_option(NO_MEMLOCK "Do not lock buffer memory (useful when running in Docker)" OFF)
ktk_tristate_option(SUPERDEBUG "Enable extra debugging with no optimisation" OFF)
ktk_tristate_option(SANITIZE "Enable clang sanitizers for testing" OFF)
ktk_tristate_option(USE_HIP "Build HIP GPU Framework" OFF)
ktk_tristate_option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
ktk_tristate_option(
    WITH_TESTS "Build and link lib/testing helper library (does not build Boost unit tests)" OFF)
ktk_tristate_option(WITH_BOOST_TESTS "Compile boost C++ unit tests" OFF)
