# Set the version of CMake required for *all* CMake files in the project. 3.8.2 to match C++17.
cmake_minimum_required(VERSION 3.8.2)
project(
    KOTEKAN
    DESCRIPTION "A high performance radio data processing pipeline"
    LANGUAGES C CXX)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

# Require at least c++17 support from the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# this makes sure we use -std=c++17, not -std=gnu++17
set(CMAKE_CXX_EXTENSIONS OFF)

# This quiets a noisy cmake warning on newer versions, triggered by some of our CMake/FindXYZ.cmake
# scripts.
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.12.0")
    cmake_policy(SET CMP0075 NEW)
endif()

# optional modules for GPUs and the like
option(USE_AIRSPY "Build Airspy Producer" OFF)
option(USE_FFTW "Build with FFTW F-engine" OFF)
option(USE_LAPACK "Build with LAPACK Linear Algebra (OpenBLAS)" OFF)
option(USE_HCC "Build HCC GPU Framework" OFF)
option(USE_HSA "Build HSA GPU Framework" OFF)
option(USE_CLOC "Use the CL offline compiler" OFF)
option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
option(USE_CUDA "Build CUDA GPU Framework" OFF)
option(USE_DPDK "Enable DPDK Framework" OFF)
option(USE_HDF5 "Build HDF5 output stages" OFF)
option(USE_OMP "Enable OpenMP" OFF)
option(USE_OLD_ROCM "Build for ROCm versions 2.3 or older" OFF)
option(NO_MEMLOCK "Do not lock buffer memory (useful when running in Docker)" OFF)
option(SUPERDEBUG "Enable extra debugging with no optimisation" OFF)
option(SANITIZE "Enable clang sanitizers for testing" OFF)
option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)
option(BOOST_TESTS "Compile Boost C++ unit tests" OFF)
option(_GOSSEC "Set buffer status messages to DEBUG log level" OFF)
option(IWYU "Enable include-what-you-use and print suggestions to stderr" OFF)
option(CCACHE "Use ccache to speed up the build" OFF)

if(${CCACHE})
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message("Using ccache from " ${CCACHE_PROGRAM})
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message("Unable to find ccache")
    endif()
endif()

if(${_GOSSEC})
    add_definitions(-D_GOSSEC=1)
    message("_GOSSEC enabled: Set buffer status messages to DEBUG log level")
endif()

if(${USE_HCC} AND (CMAKE_CXX_COMPILER MATCHES ".*hcc"))
    find_package(HCC)
    set(USE_HCC ${HCC_fOUND})
else()
    set(USE_HCC OFF)
endif()

if(${USE_HSA})
    find_package(HSA REQUIRED)
    set(USE_HSA ${HSA_FOUND})
    find_library(hsa-runtime64 /opt/rocm/hsa/lib/)
    if(${USE_OLD_ROCM})
        add_definitions(-DUSE_OLD_ROCM)
    endif()
endif()

if(${USE_OPENCL})
    set(ENV{AMDAPPSDKROOT} /opt/rocm/opencl)
    find_package(OpenCL REQUIRED)
    set(USE_OPENCL ${OPENCL_FOUND})
endif()

if(${USE_CUDA})
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    find_package(CUDA REQUIRED)
    set(USE_CUDA ${CUDA_FOUND})
    enable_language(CUDA)
endif()

if(${USE_AIRSPY})
    find_package(LIBAIRSPY)
    set(USE_AIRSPY ${LIBAIRSPY_FOUND})
endif()

if(${USE_FFTW})
    find_package(FFTW)
    set(USE_FFTW ${FFTW_FOUND})
    add_definitions(-DWITH_FFTW)
    include_directories(${FFTW_INCLUDES})
endif()

if(NOT DEFINED ARCH)
    set(ARCH "native")
endif()

include(CheckIncludeFileCXX)
if(${USE_LAPACK})

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    find_path(
        BLAS_INCLUDE_DIRS cblas.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES openblas)
    message("Using BLAS ${BLAS_LIBRARIES}")
    message("Using BLAS includes ${BLAS_INCLUDE_DIRS}")
    find_package(LAPACK REQUIRED)
    message("Using LAPACK ${LAPACK_LIBRARIES}")
    find_package(LAPACKE REQUIRED)
    message("Using LAPACKE ${LAPACKE_LIBRARIES}")

    # Check Blaze is installed
    if(DEFINED BLAZE_PATH)
        if(NOT EXISTS ${BLAZE_PATH}/blaze/Blaze.h)
            message(FATAL_ERROR "Could not find Blaze headers")
        else()
            include_directories(SYSTEM ${BLAZE_PATH})
        endif()
    endif()
    add_definitions(-DBLAZE_BLAS_MODE=1)
    add_definitions(-DBLAZE_BLAS_IS_PARALLEL=1)
endif()

# Libnuma is optional for some setups
find_package(NUMA)
if(${NUMA_FOUND})
    add_definitions(-DWITH_NUMA)
    message("NUMA support enabled")
    link_libraries(${NUMA_LIBRARY})
else()
    message("libnuma not found, NUMA support will be disabled")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-DMAC_OSX)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# Create custom build type for testing: no debug info, but asserts
string(REGEX REPLACE "( -DNDEBUG$|-DNDEBUG )" "" CMAKE_CXX_FLAGS_TEST "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "( -DNDEBUG$|-DNDEBUG )" "" CMAKE_C_FLAGS_TEST "${CMAKE_C_FLAGS_RELEASE}")

# Enable debug logging for Debug and Test builds
if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES Test)
    add_definitions(-DDEBUGGING)
    message("DEBUG logging enabled")
    message("Asserts enabled")
endif()

# include-what-you-use: this has to be set before any targets are added
if(IWYU)
    find_program(IWYU_PATH NAMES include-what-you-use iwyu)
    if(NOT IWYU_PATH)
        message(FATAL_ERROR "Could not find the program include-what-you-use")
    endif()
    if(NOT IWYU_MAPPING_FILE)
        set(IWYU_MAPPING_FILE "${KOTEKAN_SOURCE_DIR}/iwyu.kotekan.imp")
    endif()
    message("IWYU enabled: Using iwyu from ${IWYU_PATH} and mapping file ${IWYU_MAPPING_FILE}")
    execute_process(
        COMMAND ${IWYU_PATH} "--version"
        OUTPUT_VARIABLE IWYU_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message("IWYU version: ${IWYU_VERSION}")
    set(IWYU_PATH_AND_OPTIONS
        ${IWYU_PATH}
        -Xiwyu
        --max_line_length=100
        -Xiwyu
        --mapping_file=${IWYU_MAPPING_FILE}
        -Xiwyu
        --no_fwd_decls)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
    set(CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
endif()

set(GPU_MODULES "")
if(${USE_HCC})
    include_directories(${KOTEKAN_SOURCE_DIR}/lib/hcc)
    add_definitions(-DWITH_HCC)
    set(GPU_MODULES ${GPU_MODULES} "HCC ")
endif()
if(${USE_HSA})
    include_directories(${KOTEKAN_SOURCE_DIR}/lib/hsa)
    link_directories(/opt/rocm/hsa/lib/)
    add_definitions(-DWITH_HSA)
    set(GPU_MODULES ${GPU_MODULES} "HSA ")
endif()
if(${USE_OPENCL})
    include_directories(${KOTEKAN_SOURCE_DIR}/lib/opencl)
    include_directories(${OPENCL_INCLUDE_DIRS})
    add_definitions(-DWITH_OPENCL)
    set(GPU_MODULES ${GPU_MODULES} "OpenCL ")
endif()
if(${USE_CUDA})
    include_directories(${KOTEKAN_SOURCE_DIR}/lib/cuda)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
    add_definitions(-DWITH_CUDA)
    set(GPU_MODULES ${GPU_MODULES} "CUDA ")
endif()
message("GPU Modules Included: " ${GPU_MODULES})

set(INPUT_MODULES "")
if(${USE_DPDK})
    set(INPUT_MODULES ${INPUT_MODULES} "DPDK ")
    find_package(DPDK REQUIRED)
endif()
if(${USE_AIRSPY})
    add_definitions(-DWITH_AIRSPY)
    set(INPUT_MODULES ${INPUT_MODULES} "AIRSPY ")
endif()
message("Input Modules Included: " ${INPUT_MODULES})

if(${USE_HDF5})
    # only clone highfive if HIGHFIVE_PATH is not specified nor cached
    if(NOT DEFINED HIGHFIVE_PATH)
        message("HIGHFIVE_PATH not specified: Cloning HighFive repository")
        include(ExternalProject)
        find_package(Git REQUIRED)
        ExternalProject_Add(
            highfive
            PREFIX ${CMAKE_BINARY_DIR}/ext
            GIT_REPOSITORY https://github.com/jrs65/HighFive.git
            GIT_TAG extensible-datasets
            TIMEOUT 10
            UPDATE_COMMAND ${GIT_EXECUTABLE} pull
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            LOG_DOWNLOAD ON)
        ExternalProject_Get_Property(highfive source_dir)
        set(HIGHFIVE_PATH ${source_dir})
    else()
        # just to satisfy dependencies (dependencies on highfive are needed to make sure it is
        # cloned before attempting to build kotekan)
        add_custom_target(highfive COMMENT "Highfive path supplied manually.")
    endif()
    message("HDF5 enabled; using HighFive from: " ${HIGHFIVE_PATH})
    include_directories(SYSTEM ${HIGHFIVE_PATH}/include)
    find_package(HDF5 REQUIRED)
    add_definitions(-DWITH_HDF5)
endif()

# Optionally include OpenSSL for hash function support
find_package(OpenSSL)
if(OPENSSL_FOUND)
    add_definitions(-DWITH_SSL)
    set(USE_OPENSSL ON)
    include_directories(${OPENSSL_INCLUDE_DIR})
endif()

find_package(Threads REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wzero-as-null-pointer-constant")

add_compile_options(-D_GNU_SOURCE -march=${ARCH} -mtune=${ARCH} -I/opt/rocm/include)

# optimization
add_compile_options(-O3)

# lots of warnings and all warnings as errors
add_compile_options(-Wall -Wextra -Werror)

# OpenMP flag
if(${USE_OMP})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb -O2")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# dpdk uses the register keyword that got removed in C++17 because it wasn't doing anything TODO
# remove this line when dpdk people fix this:
# https://mails.dpdk.org/archives/dev/2018-July/107950.html
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register")

# Warning about missing override is called differently in clang and gcc TODO: enable once we are
# using a release version of HighFive if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Winconsistent-missing-override") else()
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wsuggest-override") endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb -O2")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

set(CMAKE_INSTALL_PREFIX "/")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # On MacOS turn off ASLR for better debugging/profiling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wl,-no_pie")
else()
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -L/opt/rocm/lib -lm")
endif()

# Add link to atomic operations library if we are using Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic")
endif()

# Add fcx-limited-range to speed up complex arithmetic if we are using GCC Add atomic operations
# library which is sometimes needed for GCC
# Disable complex math NaN/INFO range checking for performance
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fcx-limited-range HAVE_CX_LIMITED_RANGE)
if(HAVE_CX_LIMITED_RANGE AND NOT ${IWYU})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-limited-range")
endif()

include(CheckCCompilerFlag)
check_c_compiler_flag(-fcx-limited-range HAVE_C_LIMITED_RANGE)
if(HAVE_C_LIMITED_RANGE AND NOT ${IWYU})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcx-limited-range")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -latomic")
endif()

if(${USE_OPENCL})
    install(DIRECTORY lib/opencl/kernels DESTINATION /var/lib/kotekan/opencl)
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")

if(${COMPILE_DOCS})
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

# Turn off memlocking of buffers. This is useful when running in restricted environments e.g. Docker
# containers
if(${NO_MEMLOCK})
    message("Do not lock buffer memory.")
    add_definitions(-DWITH_NO_MEMLOCK)
endif()

# Improve debugging by turning off all optimisations.
if(${SUPERDEBUG})
    message("Superdebugging enabled!!")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
endif()

# Turn on sanitizers for finding memory issues.
if(${SANITIZE})
    message("Sanitization enabled!!")
    set(CMAKE_C_FLAGS_DEBUG
        string ("${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls "
                "-fsanitize=address"))
    set(CMAKE_CXX_FLAGS_DEBUG
        string ("${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls
            -fsanitize=address"))
endif()

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(scripts)
add_subdirectory(external)
add_subdirectory(config)

# Testing with boost
if(${BOOST_TESTS})
    set(_BOOST_TESTS_DIR
        tests/boost/
        CACHE INTERNAL "Path to boost test sources")
    message("BOOST_TESTS enabled: building tests in " ${_BOOST_TESTS_DIR})
    add_subdirectory(${_BOOST_TESTS_DIR})
else()
    message("BOOST_TESTS disabled")
endif()

# Enforce code formatting rules with clang-format.
find_program(
    CLANG_FORMAT_PATH
    NAMES "clang-format-8" "clang-format"
    DOC "Path to clang-format executable")
if(NOT CLANG_FORMAT_PATH)
    message(STATUS "clang-format not found.")
else()
    message(STATUS "clang-format found: ${CLANG_FORMAT_PATH}")
    set(DO_CLANG_FORMAT "${CLANG_FORMAT_PATH}" "-i -style=file")
endif()

# Remove files containing substring from a list of file names.
function(exclude_files_containing var exclude_string)
    set(listVar "")
    foreach(file ${ARGN})
        string(FIND ${file} ${exclude_string} EXCLUDE_STRING_FOUND)
        if(${EXCLUDE_STRING_FOUND} EQUAL -1)
            list(APPEND listVar ${file})
        endif()
    endforeach()
    set(${var}
        "${listVar}"
        PARENT_SCOPE)
endfunction()

if(CLANG_FORMAT_PATH)
    # Find all .c(pp) and .h(pp) files.
    file(GLOB_RECURSE KOTEKAN_ALL_CPP_FILES *.cpp)
    file(GLOB_RECURSE KOTEKAN_ALL_HPP_FILES *.hpp)
    file(GLOB_RECURSE KOTEKAN_ALL_C_FILES *.c)
    file(GLOB_RECURSE KOTEKAN_ALL_H_FILES *.h)

    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_CPP_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_HPP_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_C_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_H_FILES})

    # Exclude all code we don't want auto-formatted (external libs).
    set(EXCLUDE_DIR "external")
    exclude_files_containing(FILES_TO_FORMAT ${EXCLUDE_DIR} ${FILES_TO_FORMAT})
    set(EXCLUDE_DIR "build")
    exclude_files_containing(FILES_TO_FORMAT ${EXCLUDE_DIR} ${FILES_TO_FORMAT})

    # Use .clang-format file to auto format.
    add_custom_target(
        clang-format
        COMMAND ${CLANG_FORMAT_PATH} -style=file -i ${FILES_TO_FORMAT}
        COMMENT "Running clang-format")
endif()
