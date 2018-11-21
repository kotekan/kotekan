cmake_minimum_required (VERSION 2.8)
project (KOTEKAN)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

# This is required early on for some versions of GCC, do not remove.
add_definitions(/D__STDC_FORMAT_MACROS)

# This quiets a noisy cmake warning on newer versions,
# triggered by some of our CMake/FindXYZ.cmake scripts.
if(NOT ${CMAKE_VERSION} VERSION_LESS "3.12.0")
    cmake_policy(SET CMP0075 NEW)
endif ()

#optional modules for GPUs and the like
option(USE_AIRSPY "Build Airspy Producer" OFF)
option(USE_FFTW "Build with FFTW F-engine" OFF)
option(USE_LAPACK "Build with LAPACK Linear Algebra (OpenBLAS)" OFF)
option(USE_HCC "Build HCC GPU Framework" OFF)
option(USE_HSA "Build HSA GPU Framework" OFF)
option(USE_PRECOMPILED_OPENCL "Use precompiled OpenCL binaries" OFF)
option(USE_CLOC "Use the CL offline compiler" OFF)
option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
option(USE_DPDK "Enable DPDK Framework" OFF)
option(USE_HDF5 "Build HDF5 output processes" OFF)
option(USE_OMP "Enable OpenMP" OFF)
option(SUPERDEBUG "Enable extra debugging with no optimisation" OFF)
option(SANITIZE "Enable clang sanitizers for testing" OFF)
option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)
option(BOOST_TESTS "Compile Boost C++ unit tests" OFF)
option(_GOSSEC "Set buffer status messages to DEBUG log level" OFF)

if(${_GOSSEC})
    add_definitions(-D_GOSSEC=1)
    MESSAGE("_GOSSEC enabled: Set buffer status messages to DEBUG log level")
endif()

if (${USE_HCC} AND (CMAKE_CXX_COMPILER MATCHES ".*hcc"))
  find_package( HCC )
  set(USE_HCC ${HCC_fOUND})
else ()
  set(USE_HCC OFF)
endif ()

if (${USE_HSA})
  find_package( HSA REQUIRED )
  set(USE_HSA ${HSA_FOUND})
  find_library( hsa-runtime64 /opt/rocm/hsa/lib/ )
endif ()

if (${USE_OPENCL})
  set( ENV{AMDAPPSDKROOT} /opt/rocm/opencl)
  find_package( OpenCL REQUIRED )
  set(USE_OPENCL ${OPENCL_FOUND})
endif ()

if (${USE_AIRSPY})
  find_package( LibAirSpy )
  set(USE_AIRSPY ${LIBAIRSPY_FOUND})
endif ()

if (${USE_FFTW})
  find_package( FFTW)
  set(USE_FFTW ${FFTW_FOUND})
  add_definitions(-DWITH_FFTW)
endif()

if (${USE_LAPACK})
  if (DEFINED OPENBLAS_PATH)
    if (NOT EXISTS ${OPENBLAS_PATH})
      MESSAGE( "OPENBLAS_PATH: \"${OPENBLAS_PATH}\" does not exsist" )
    endif()
    set(ENV{CMAKE_PREFIX_PATH} ${OPENBLAS_PATH})
  endif()
  find_package( OpenBLAS REQUIRED )
endif()

#Libevent is required now
find_package(LibEvent REQUIRED )
include_directories(${EVENT_INCLUDE_DIR}/../)
link_directories(${EVENT_LIBRARY_DIR})

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_definitions(-DMAC_OSX)
else()
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUGGING)
endif()

set(GPU_MODULES "")
if (${USE_HCC})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/hcc)
  add_definitions(-DWITH_HCC)
  set(GPU_MODULES ${GPU_MODULES} "HCC ")
endif ()
if (${USE_HSA})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
  link_directories ( /opt/rocm/hsa/lib/ )
  add_definitions(-DWITH_HSA)
  set(GPU_MODULES ${GPU_MODULES} "HSA ")
endif ()
if (${USE_OPENCL})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
  include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  add_definitions(-DWITH_OPENCL)
  set(GPU_MODULES ${GPU_MODULES} "OpenCL")
endif ()
MESSAGE("GPU Modules Included: " ${GPU_MODULES})

set(INPUT_MODULES "")
if (${USE_DPDK})
  set(INPUT_MODULES ${INPUT_MODULES} "DPDK ")
endif ()
if (${USE_AIRSPY})
  add_definitions(-DWITH_AIRSPY)
  set(INPUT_MODULES ${INPUT_MODULES} "AIRSPY ")
endif ()
MESSAGE("Input Modules Included: " ${INPUT_MODULES})

if (${USE_HDF5})
    # only clone highfive if HIGHFIVE_PATH is not specified nor cached
    if (NOT DEFINED HIGHFIVE_PATH)
        MESSAGE("HIGHFIVE_PATH not specified: Cloning HighFive repository")
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
            LOG_DOWNLOAD ON
        )
        ExternalProject_Get_Property(highfive source_dir)
        set(HIGHFIVE_PATH ${source_dir} CACHE INTERNAL "Path to HighFive")
    else()
        # just to satisfy dependencies (dependencies on highfive are needed
        # to make sure it is cloned before attempting to build kotekan)
        add_custom_target(highfive)
    endif()
    MESSAGE("HDF5 enabled; using HighFive from: " ${HIGHFIVE_PATH})
    include_directories ( ${HIGHFIVE_PATH}/include )
    find_package(HDF5 REQUIRED)
    add_definitions(-DWITH_HDF5)
endif ()

# Optionally include OpenSSL for hash function support
find_package( OpenSSL )
if( OPENSSL_FOUND )
    add_definitions(-DWITH_SSL)
    set (USE_OPENSSL ON)
    include_directories(${OPENSSL_INCLUDE_DIR})
endif()

find_package(Threads REQUIRED)

include_directories (${KOTEKAN_SOURCE_DIR}/include)

# Note that -lrt here is needed on some versions of CentOS
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt " )
endif ()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -Wextra -Werror -march=native -mtune=native -I/opt/rocm/include")

# OpenMP flag
if(${USE_OMP})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb -O2")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Note that -lrt here is needed on some versions of CentOS
if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt " )
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -Wall -Wextra -Werror -march=native -mtune=native -I/opt/rocm/include")

# Warning about missing override is called differently in clang and gcc
# TODO: enable once we are using a release version of HighFive
#if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Winconsistent-missing-override")
#else()
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wsuggest-override")
#endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb -O2")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

SET(CMAKE_INSTALL_PREFIX "/")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # On MacOS turn off ASLR for better debugging/profiling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wl,-no_pie")
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -L/opt/rocm/lib -lm")
endif()

# Add fcx-limited-range to speed up complex arithmetic if we are using GCC
# Add atomic operations library which is sometimes needed for GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcx-limited-range")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-limited-range")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -latomic")
endif()

if (${USE_OPENCL})
    install(DIRECTORY lib/opencl/kernels DESTINATION /var/lib/kotekan/opencl )
endif ()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

if (${COMPILE_DOCS})
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()


# Improve debugging by turning off all optimisations.
if (${SUPERDEBUG})
    MESSAGE("Superdebugging enabled!!")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer ")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
endif()

# Turn on sanitizers for finding memory issues.
if (${SANITIZE})
    MESSAGE("Sanitization enabled!!")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address")
endif()


add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(include)

# Testing with boost
if (${BOOST_TESTS})
    set(BOOST_TESTS_DIR tests/boost/ CACHE INTERNAL "Path to boost test sources")
    MESSAGE("BOOST_TESTS enabled: building tests in " ${BOOST_TESTS_DIR})
    add_subdirectory(${BOOST_TESTS_DIR})
else()
    MESSAGE("BOOST_TESTS disabled")
endif()
