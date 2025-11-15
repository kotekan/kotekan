# Final feature summary table.
#
# Developer note: when adding new configure options or feature toggles, update the flag lists below
# so the summary remains complete. This file should just be informative, not affect the build.

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/Color.cmake)

# Column widths for aligned summary output
set(_kfeature_name_col_width 24)
set(_kfeature_state_col_width 8)

# kpad_right: pad a string on the right with spaces up to a fixed width
function(kpad_right out_var in_str width)
    string(LENGTH "${in_str}" len)
    set(res "${in_str}")
    math(EXPR pad "${width} - ${len}")
    if(pad GREATER 0)
        set(spaces "")
        while(pad GREATER 0)
            set(spaces "${spaces} ")
            math(EXPR pad "${pad} - 1")
        endwhile()
        set(res "${res}${spaces}")
    endif()
    set(${out_var}
        "${res}"
        PARENT_SCOPE)
endfunction()

# kfeature_row: print one aligned feature row; optional 5th arg is toggle var (e.g., USE_CUDA)
function(kfeature_row name enabled reason present)
    set(toggle_var "")
    if(ARGC GREATER 4)
        set(toggle_var "${ARGV4}")
    endif()
    
    # Determine the effective enabled state and the display token
    set(_kfeat_enabled FALSE)
    string(TOUPPER "${enabled}" _kfeat_enabled_upper)
    set(_kfeat_true "ON;TRUE;YES;Y;1")
    set(_kfeat_false "OFF;FALSE;NO;N;0")
    if(_kfeat_enabled_upper IN_LIST _kfeat_true)
        set(_kfeat_enabled TRUE)
    elseif(_kfeat_enabled_upper IN_LIST _kfeat_false OR _kfeat_enabled_upper STREQUAL "")
        set(_kfeat_enabled FALSE)
    endif()

    string(TOLOWER "${reason}" _kfeat_reason_lc)
    set(_kfeat_reason_state "")
    if(_kfeat_reason_lc MATCHES "auto-?detected|enabled|found")
        set(_kfeat_reason_state TRUE)
    endif()
    if(_kfeat_reason_lc MATCHES "disabled|not found|not available|not runnable|compiler not")
        set(_kfeat_reason_state FALSE)
    endif()
    if(NOT "${_kfeat_reason_state}" STREQUAL "")
        set(_kfeat_enabled ${_kfeat_reason_state})
    endif()

    set(state "")
    set(_kfeat_toggle_upper "")
    if(NOT "${toggle_var}" STREQUAL "")
        get_property(_kfeat_toggle CACHE ${toggle_var} PROPERTY VALUE)
        if(DEFINED _kfeat_toggle)
            string(TOUPPER "${_kfeat_toggle}" _kfeat_toggle_upper)
        endif()
    endif()
    if(_kfeat_toggle_upper STREQUAL "AUTO")
        if(_kfeat_enabled)
            set(state "AUTO-ON")
        else()
            set(state "AUTO-OFF")
        endif()
    elseif(_kfeat_toggle_upper STREQUAL "ON" OR _kfeat_toggle_upper STREQUAL "OFF")
        set(state "${_kfeat_toggle_upper}")
    else()
        if(_kfeat_enabled)
            set(state "ON")
        else()
            set(state "OFF")
        endif()
    endif()

    set(color "${KTK_YELLOW}")
    if(state MATCHES "^AUTO")
        if(_kfeat_enabled)
            set(color "${KTK_GREEN}")
        else()
            set(color "${KTK_YELLOW}")
        endif()
    elseif(_kfeat_enabled)
        set(color "${KTK_GREEN}")
    else()
        if(present)
            set(color "${KTK_YELLOW}")
        elseif(_kfeat_reason_lc MATCHES "missing|not found|not runnable|not available|compiler not")
            set(color "${KTK_RED}")
        else()
            set(color "${KTK_YELLOW}")
        endif()
    endif()

    # Pad columns so that the parenthetical notes align
    set(name_col "${name}:")
    kpad_right(name_col "${name_col}" ${_kfeature_name_col_width})
    kpad_right(state_col "${state}" ${_kfeature_state_col_width})

    set(toggle_note "")
    if(NOT "${toggle_var}" STREQUAL "")
        # If the toggle is a cached BOOL option, show the explicit command-line form with =ON/=OFF.
        # Otherwise, just show -D<VAR>.
        get_property(
            _ktk_toggle_type
            CACHE ${toggle_var}
            PROPERTY TYPE)
        if("${_ktk_toggle_type}" STREQUAL "BOOL")
            if(_kfeat_toggle_upper STREQUAL "AUTO")
                set(toggle_note ", -D${toggle_var}=ON|OFF")
            elseif(_kfeat_enabled)
                set(toggle_note ", -D${toggle_var}=ON")
            else()
                set(toggle_note ", -D${toggle_var}=OFF")
            endif()
        else()
            set(toggle_note ", -D${toggle_var}")
        endif()
    endif()
    message(STATUS "${color}${name_col} ${state_col} (${reason}${toggle_note})${KTK_RESET}")
endfunction()

# kfeature_header: print a section header in the summary
function(kfeature_header title)
    kmsg_status("")
    kmsg_status("==== ${title} ====")
endfunction()

# Key/Value printer for build metadata kfeature_kv: print a key/value line in the summary
function(kfeature_kv name value)
    kmsg_status("${name}: ${value}")
endfunction()

# Check availability of various features Then later on, print it.

set(_asdf_present OFF)
if("${USE_ASDF}" STREQUAL "OFF")
    find_package(asdf-cxx QUIET)
    set(_asdf_present ${ASDF_CXX_FOUND})
endif()

set(_gdal_present OFF)
if("${USE_GDAL}" STREQUAL "OFF")
    find_package(GDAL QUIET)
    set(_gdal_present ${GDAL_FOUND})
endif()

set(_hdf5_present OFF)
if("${USE_HDF5}" STREQUAL "OFF")
    find_package(HDF5 QUIET)
    find_package(HighFive QUIET)
    if(HDF5_FOUND AND HighFive_FOUND)
        set(_hdf5_present ON)
    endif()
endif()

set(_fftw_present OFF)
if("${USE_FFTW}" STREQUAL "OFF")
    find_package(FFTW QUIET)
    set(_fftw_present ${FFTW_FOUND})
endif()

set(_lapack_blaze_present OFF)
if("${USE_LAPACK_BLAZE}" STREQUAL "OFF")
    find_package(LAPACKE QUIET)
    find_package(Blaze QUIET)
    if(LAPACKE_FOUND AND BLAZE_FOUND)
        set(_lapack_blaze_present ON)
    endif()
endif()

if(DEFINED KOTEKAN_AIRSPY_REASON)
    set(AIRSPY_REASON "${KOTEKAN_AIRSPY_REASON}")
else()
    set(AIRSPY_REASON "disabled")
endif()
set(_airspy_present OFF)
if(DEFINED KOTEKAN_AIRSPY_ENABLED)
    set(_airspy_present "${KOTEKAN_AIRSPY_ENABLED}")
endif()

# Checks done; now print availability of various features

# GPU
kfeature_header("GPU Features")
# Alphabetical: CUDA, HIP, OpenCL
kfeature_row("CUDA" "${USE_CUDA}" "${CUDA_REASON}" OFF USE_CUDA)
kfeature_row("HIP" "${USE_HIP}" "${HIP_REASON}" OFF USE_HIP)
kfeature_row("OpenCL" "${USE_OPENCL}" "${OPENCL_REASON}" OFF USE_OPENCL)

# IO
kfeature_header("I/O Formats")
kfeature_row("ASDF" "${USE_ASDF}" "${ASDF_REASON}" "${_asdf_present}" USE_ASDF)
kfeature_row("GDAL" "${USE_GDAL}" "${GDAL_REASON}" "${_gdal_present}" USE_GDAL)
kfeature_row("HDF5" "${USE_HDF5}" "${HDF5_REASON}" "${_hdf5_present}" USE_HDF5)

# Math
kfeature_header("Math")
kfeature_row("FFTW" "${USE_FFTW}" "${FFTW_REASON}" "${_fftw_present}" USE_FFTW)
kfeature_row("LAPACK/Blaze" "${USE_LAPACK_BLAZE}" "${LAPACK_BLAZE_REASON}"
             "${_lapack_blaze_present}" USE_LAPACK_BLAZE)

# Other
kfeature_header("Other")
# Alphabetical: Airspy, DPDK, Julia, NUMA, OpenMP, OpenSSL
kfeature_row("Airspy" "${USE_AIRSPY}" "${AIRSPY_REASON}" "${_airspy_present}" USE_AIRSPY)
# Note: When WITH_BOOST_TESTS=ON, DPDK is disabled to avoid linker issues
kfeature_row("DPDK" "${USE_DPDK}" "${DPDK_REASON}" "${USE_DPDK}" USE_DPDK)
kfeature_row("Julia" "${USE_JULIA}" "${JULIA_REASON}" "${USE_JULIA}" USE_JULIA)

# Build meta
kfeature_header("Build")

# Build Type aligned so value starts where ON/OFF would
set(_bt_name "Build Type:")
kpad_right(_bt_name "${_bt_name}" ${_kfeature_name_col_width})
kmsg_status("${_bt_name} ${CMAKE_BUILD_TYPE} (-DCMAKE_BUILD_TYPE=Debug|Release|Test)")

if(DEFINED KOTEKAN_USE_OMP_REASON)
    set(OMP_REASON "${KOTEKAN_USE_OMP_REASON}")
else()
    if("${USE_OMP}" STREQUAL "ON")
        set(OMP_REASON "enabled")
    else()
        set(OMP_REASON "disabled")
    endif()
endif()

if("${USE_OPENSSL}" STREQUAL "ON")
    set(_ossl_reason "${OPENSSL_REASON}")
else()
    # Hint: use -DOPENSSL_ROOT_DIR=<path> for non-standard installs
    set(_ossl_reason "${OPENSSL_REASON}; use -DOPENSSL_ROOT_DIR=<path> for non-standard installs")
endif()

kfeature_row("NUMA" "${USE_NUMA}" "${NUMA_REASON}" OFF USE_NUMA)
kfeature_row("OpenMP" "${USE_OMP}" "${OMP_REASON}" OFF USE_OMP)
kfeature_row("OpenSSL" "${USE_OPENSSL}" "${_ossl_reason}" OFF USE_OPENSSL)

if("${COMPILE_DOCS}" STREQUAL "ON")
    set(DOCS_REASON "enabled")
else()
    set(DOCS_REASON "disabled")
endif()
kfeature_row("Docs" "${COMPILE_DOCS}" "${DOCS_REASON}" OFF COMPILE_DOCS)

if("${WITH_TESTS}" STREQUAL "ON")
    set(TESTS_REASON "enabled: builds and links lib/testing")
else()
    set(TESTS_REASON "disabled")
endif()
kfeature_row("C++ Tests (lib)" "${WITH_TESTS}" "${TESTS_REASON}" OFF WITH_TESTS)

if("${WITH_BOOST_TESTS}" STREQUAL "ON")
    set(BOOSTTEST_REASON "enabled: builds tests/boost and disables DPDK")
else()
    set(BOOSTTEST_REASON "disabled")
endif()
kfeature_row("Boost Tests" "${WITH_BOOST_TESTS}" "${BOOSTTEST_REASON}" OFF WITH_BOOST_TESTS)

if("${NO_MEMLOCK}" STREQUAL "ON")
    set(NOMEMLOCK_REASON "enabled")
else()
    set(NOMEMLOCK_REASON "disabled")
endif()
kfeature_row("No Memlock" "${NO_MEMLOCK}" "${NOMEMLOCK_REASON}" OFF NO_MEMLOCK)

kfeature_row("ccache" "${CCACHE}" "${CCACHE_REASON}" OFF CCACHE)

set(WERROR_REASON "disabled")
if("${WERROR}" STREQUAL "ON")
    set(WERROR_REASON "enabled")
endif()
kfeature_row("Warnings-as-errors" "${WERROR}" "${WERROR_REASON}" OFF WERROR)

set(IWYU_REASON "disabled")
if("${IWYU}" STREQUAL "ON")
    set(IWYU_REASON "enabled")
endif()
kfeature_row("include-what-you-use" "${IWYU}" "${IWYU_REASON}" OFF IWYU)

# Flag overrides requested by users; show compact key=value list.

set(_ktk_flag_openssl "")
if(DEFINED OPENSSL_ROOT_DIR AND NOT "${OPENSSL_ROOT_DIR}" STREQUAL "")
    set(_ktk_flag_openssl "OPENSSL_ROOT_DIR='${OPENSSL_ROOT_DIR}'")
else()
    set(_ktk_flag_openssl "OPENSSL_ROOT_DIR=")
endif()

if(DEFINED BLAZE_PATH AND NOT "${BLAZE_PATH}" STREQUAL "")
    set(_ktk_flag_blaze "BLAZE_PATH='${BLAZE_PATH}'")
else()
    set(_ktk_flag_blaze "BLAZE_PATH=")
endif()

if(DEFINED HDF5Plugin_PLUGIN_DIR AND NOT "${HDF5Plugin_PLUGIN_DIR}" STREQUAL "")
    set(_ktk_flag_hdf5_plugin "HDF5_PLUGIN_DIR='${HDF5Plugin_PLUGIN_DIR}'")
else()
    set(_ktk_flag_hdf5_plugin "HDF5_PLUGIN_DIR=")
endif()

if(DEFINED KOTEKAN_CL_TARGET_OPENCL_VERSION AND NOT "${KOTEKAN_CL_TARGET_OPENCL_VERSION}" STREQUAL
                                                "")
    set(_ktk_flag_cl "CL_TARGET_OPENCL_VERSION=${KOTEKAN_CL_TARGET_OPENCL_VERSION}")
else()
    set(_ktk_flag_cl "CL_TARGET_OPENCL_VERSION=")
endif()

if(DEFINED CMAKE_INSTALL_PREFIX AND NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "")
    set(_ktk_flag_prefix "CMAKE_INSTALL_PREFIX='${CMAKE_INSTALL_PREFIX}'")
else()
    set(_ktk_flag_prefix "CMAKE_INSTALL_PREFIX=")
endif()

if(DEFINED CUDAToolkit_ROOT AND NOT "${CUDAToolkit_ROOT}" STREQUAL "")
    set(_ktk_flag_cuda "CUDAToolkit_ROOT='${CUDAToolkit_ROOT}'")
else()
    set(_ktk_flag_cuda "CUDAToolkit_ROOT=")
endif()

set(_ktk_flag_summary "${_ktk_flag_openssl} ${_ktk_flag_blaze} ${_ktk_flag_hdf5_plugin} "
                      "${_ktk_flag_cl} ${_ktk_flag_prefix} ${_ktk_flag_cuda}")
kmsg_status("Flag Overrides: ${_ktk_flag_summary}")
kmsg_status("Cache Hint: run 'cmake -U<VAR> .' or remove CMakeCache.txt to reset")

# Compilers
kfeature_header("Compilers")
set(_c_vers ${CMAKE_C_COMPILER_VERSION})
if(NOT _c_vers)
    set(_c_vers "unknown")
endif()
set(_cxx_vers ${CMAKE_CXX_COMPILER_VERSION})
if(NOT _cxx_vers)
    set(_cxx_vers "unknown")
endif()
kfeature_kv("C" "${CMAKE_C_COMPILER_ID} ${_c_vers} (${CMAKE_C_COMPILER})")
kfeature_kv("C++" "${CMAKE_CXX_COMPILER_ID} ${_cxx_vers} (${CMAKE_CXX_COMPILER})")
if(NOT ${USE_CUDA} STREQUAL "OFF")
    set(_cuda_vers ${CMAKE_CUDA_COMPILER_VERSION})
    if(NOT _cuda_vers)
        set(_cuda_vers "unknown")
    endif()
    kfeature_kv("CUDA" "NVCC ${_cuda_vers} (${CMAKE_CUDA_COMPILER})")
endif()

kmsg_status("")
