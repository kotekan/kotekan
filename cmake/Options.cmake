# All user-facing build options (no logic here).
#
# Developer note: when adding a new option, also update the configure summary (cmake/Summary.cmake),
# the documentation (Sphinx CMake option pages), and the version metadata
# (lib/version/version.c.in).

# _ktk_option: declare an option and reuse cached value when present
function(_ktk_option name description default_value)
    get_property(
        _ktk_cache_has
        CACHE ${name}
        PROPERTY TYPE
        SET)
    if(_ktk_cache_has)
        get_property(
            default_value
            CACHE ${name}
            PROPERTY VALUE)
    endif()
    option(${name} "${description}" ${default_value})
endfunction()

_ktk_option(USE_AIRSPY "Build Airspy Producer" ON)
_ktk_option(USE_FFTW "Build with FFTW F-engine" ON)
_ktk_option(USE_LAPACK_BLAZE "Build with LAPACK Linear Algebra (OpenBLAS) and Blaze support" ON)
_ktk_option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
_ktk_option(USE_CUDA "Build CUDA GPU Framework" ON)
_ktk_option(USE_HIP "Build HIP GPU Framework" OFF)
# DPDK selection (>=19.11 via pkg-config)
set(USE_DPDK
    "ON"
    CACHE STRING "DPDK usage: ON (auto), OFF")
set_property(CACHE USE_DPDK PROPERTY STRINGS ON OFF)
_ktk_option(USE_ASDF "Build ASDF output stages" ON)
_ktk_option(USE_GDAL "Build GDAL output stages" ON)
_ktk_option(USE_HDF5 "Build HDF5 output stages" ON)
_ktk_option(USE_JULIA "Build Julia-based features" ON)
_ktk_option(USE_OMP "Enable OpenMP" ON)

option(USE_OLD_ROCM "Build for ROCm versions 2.3 or older" OFF)

_ktk_option(USE_NUMA "Enable NUMA support in core (libnuma)" ON)
_ktk_option(USE_OPENSSL "Enable OpenSSL (hash) support in core" ON)
option(NO_MEMLOCK "Do not lock buffer memory (useful when running in Docker)" OFF)
option(SUPERDEBUG "Enable extra debugging with no optimisation" OFF)
option(SANITIZE "Enable clang sanitizers for testing" OFF)
option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)
option(WITH_TESTS "Build and link lib/testing helper library (does not build Boost unit tests)" OFF)
option(WITH_BOOST_TESTS "Compile boost C++ unit tests" OFF)
option(IWYU "Enable include-what-you-use and print suggestions to stderr" OFF)
option(CCACHE "Use ccache to speed up the build" OFF)
option(WERROR "Warnings are errors" ON)
option(CMAKE_LINK_WHAT_YOU_USE "Report missing link dependencies while building" OFF)
