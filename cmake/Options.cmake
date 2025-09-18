# All user-facing build options (no logic here).
#
# Developer note: when adding a new option, also update the configure summary (cmake/Summary.cmake),
# the documentation (Sphinx CMake option pages), and the version metadata
# (lib/version/version.c.in).


# ktk_tristate_option(<FLAG> <DESC> <DEFAULT> [EXPORTS] [PREFIX <name>])
#
# - <FLAG>    : cache variable name (e.g., USE_ASDF)
# - <DESC>    : help text
# - <DEFAULT> : one of AUTO, ON, OFF  (case-insensitive)
#
# Optional:
#   EXPORTS         -> initializes <PREFIX>_ENABLED and <PREFIX>_REASON in PARENT_SCOPE
#   PREFIX <name>   -> export prefix (defaults to <FLAG>)
#
# Notes:
# - Does not overwrite an existing cache value (no FORCE).
# - Sets allowed values list in cmake-gui/ccmake.

# ktk_tristate_option(<FLAG> <DESC> <DEFAULT> [EXPORTS] [PREFIX <name>])
#
# - <DEFAULT>: one of AUTO, ON, OFF (case-insensitive; boolean synonyms OK)
# - Accepts user -D<FLAG>=<value> with common CMake boolean synonyms:
#     ON:  ON TRUE YES Y 1
#     OFF: OFF FALSE NO  N 0
#     AUTO: AUTO
# - Canonicalizes the cache entry to one of: AUTO / ON / OFF
#
function(ktk_tristate_option FLAG DESC DEFAULT)
  set(options EXPORTS)
  set(oneValueArgs PREFIX)
  cmake_parse_arguments(KTK "${options}" "${oneValueArgs}" "" ${ARGN})

  # Normalization helper
  function(_ktk_norm IN OUTVAR)
    string(STRIP "${IN}" _s)
    string(TOUPPER "${_s}" _u)
    set(_on  "ON;TRUE;YES;Y;1")
    set(_off "OFF;FALSE;NO;N;0")
    if(_u STREQUAL "AUTO")
      set(${OUTVAR} "AUTO" PARENT_SCOPE)
    elseif(_u IN_LIST _on)
      set(${OUTVAR} "ON"   PARENT_SCOPE)
    elseif(_u IN_LIST _off)
      set(${OUTVAR} "OFF"  PARENT_SCOPE)
    else()
      message(FATAL_ERROR
        "Invalid value '${IN}' for ${FLAG}. Use one of: AUTO, ON, OFF "
        "(ON aliases: ON/TRUE/YES/Y/1; OFF aliases: OFF/FALSE/NO/N/0).")
    endif()
  endfunction()

  # Normalize default and ensure cache entry exists
  _ktk_norm("${DEFAULT}" _def)
  if(NOT DEFINED CACHE{${FLAG}})
    set(${FLAG} "${_def}" CACHE STRING "${DESC} (AUTO/ON/OFF)")
  endif()

  # Validate/normalize existing user-provided value, then canonicalize in cache
  get_property(_cur CACHE ${FLAG} PROPERTY VALUE)
  _ktk_norm("${_cur}" _canon)
  if(NOT _canon STREQUAL "${_cur}")
    # Canonicalize to AUTO/ON/OFF in cache so GUIs and scripts see the clean form
    set(${FLAG} "${_canon}" CACHE STRING "${DESC} (AUTO/ON/OFF)" FORCE)
  endif()

  # Present choices in GUIs
  set_property(CACHE ${FLAG} PROPERTY STRINGS AUTO ON OFF)
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
ktk_tristate_option(USE_LAPACK_BLAZE "Build with LAPACK Linear Algebra (OpenBLAS) and Blaze support" AUTO)
ktk_tristate_option(USE_OMP "Enable OpenMP" AUTO)
ktk_tristate_option(USE_OPENSSL "Enable OpenSSL (hash) support in core" AUTO)

# ON by default
ktk_tristate_option(USE_NUMA "Enable NUMA support in core (libnuma)" ON)
ktk_tristate_option(WERROR "Warnings are errors" ON)

# OFF by default
ktk_tristate_option(CCACHE "Use ccache to speed up the build" OFF)
ktk_tristate_option(CMAKE_LINK_WHAT_YOU_USE "Report missing link dependencies while building" OFF)
ktk_tristate_option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)
ktk_tristate_option(IWYU "Enable include-what-you-use and print suggestions to stderr" OFF)
ktk_tristate_option(NO_MEMLOCK "Do not lock buffer memory (useful when running in Docker)" OFF)
ktk_tristate_option(SUPERDEBUG "Enable extra debugging with no optimisation" OFF)
ktk_tristate_option(SANITIZE "Enable clang sanitizers for testing" OFF)
ktk_tristate_option(USE_HIP "Build HIP GPU Framework" OFF)
ktk_tristate_option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
ktk_tristate_option(WITH_TESTS "Build and link lib/testing helper library (does not build Boost unit tests)" OFF)
ktk_tristate_option(WITH_BOOST_TESTS "Compile boost C++ unit tests" OFF)
