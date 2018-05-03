cmake_minimum_required (VERSION 2.8)
project (KOTEKAN)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

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
option(DPDK_VDIF "Use DPDK VDIF settings" OFF)
option(USE_HDF5 "Build HDF5 output processes" OFF)
option(USE_LAPACK "Include OpenBLAS/LAPACK" OFF)
option(COMPILE_DOCS "Use Sphinx to compile documentation" OFF)

if (${USE_HCC} AND (CMAKE_CXX_COMPILER MATCHES ".*hcc"))
  find_package( HCC )
  set(USE_HCC ${HCC_fOUND})
else ()
  set(USE_HCC OFF)
endif ()
if (${USE_HSA})
  find_package( HSA )
  set(USE_HSA ${HSA_FOUND})
  find_library( hsa-runtime64 /opt/rocm/hsa/lib/ )
endif ()
if (${USE_OPENCL})
  find_package( OpenCL )
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
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lfftw3f ")
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

if(${DPDK_VDIF})
  add_definitions(-DDPDK_VDIF_MODE)
endif()


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_definitions(-DMAC_OSX)
  add_definitions(-mavx2)
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
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lhsa-runtime64 ")
  add_definitions(-DWITH_HSA)
  set(GPU_MODULES ${GPU_MODULES} "HSA ")
endif ()
if (${USE_OPENCL})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
  include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL ")
  add_definitions(-DWITH_OPENCL)
  set(GPU_MODULES ${GPU_MODULES} "OpenCL")
endif ()
MESSAGE("GPU Modules Included: " ${GPU_MODULES})

set(INPUT_MODULES "")
if (${USE_DPDK})
  set(INPUT_MODULES ${INPUT_MODULES} "DPDK ")
endif ()
if (${USE_AIRSPY})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lairspy ")
  add_definitions(-DWITH_AIRSPY)
  set(INPUT_MODULES ${INPUT_MODULES} "AIRSPY ")
endif ()
MESSAGE("Input Modules Included: " ${INPUT_MODULES})


if (${USE_DPDK})
  # DPDK Includes
  include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
  link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )

  set (DPDK_LIST
          rte_ethdev   rte_distributor  rte_kni     rte_mbuf      rte_pmd_bond   rte_pmd_ring
          rte_power    rte_timer        rte_ring    rte_sched     rte_table
          rte_acl      rte_eal          rte_kvargs  rte_mempool   rte_pmd_e1000
          rte_cfgfile  rte_hash         rte_lpm     rte_meter     rte_pmd_i40e   rte_pmd_vmxnet3_uio
          rte_cmdline  rte_ip_frag      rte_pipeline  rte_pmd_ixgbe  rte_port
  )
  set (DPDK_LIBS
        -Wl,--start-group ${DPDK_LIST} -Wl,--end-group -ldl )
  add_definitions(-DWITH_DPDK)
  MESSAGE("DPDK Enabled")
else ()
  MESSAGE("DPDK Disabled")
endif ()

if (${USE_HDF5})
  # HDF5 setup
  include_directories ( ${HIGHFIVE_PATH}/include )
  find_package(HDF5 REQUIRED)
  add_definitions(-DWITH_HDF5)
  MESSAGE("HDF5 enabled; using HighFive from: " ${HIGHFIVE_PATH})
else ()
  MESSAGE("HDF5 disabled")
endif ()

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include the version string
#add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
#add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")
configure_file(
  ${CMAKE_SOURCE_DIR}/lib/version.h.in
  ${CMAKE_BINARY_DIR}/generated/version.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)

find_package(Threads REQUIRED)

include_directories (${KOTEKAN_SOURCE_DIR}/include)
#include_directories (${KOTEKAN_SOURCE_DIR}/lib/sim_output)

# Required for some SSE/AVX intrinsics
add_definitions(-mssse3 -D__STDC_LIMIT_MACROS)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -march=native -mtune=native -I/opt/rocm/include")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb -O2")  # Reduce the optimisation for debugging

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Note C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -Wall -march=native -mtune=native -fopenmp -I/opt/rocm/include")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb -O2")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

SET(CMAKE_INSTALL_PREFIX "/")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # On MacOS turn off ASLR for better debugging/profiling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wl,-no_pie")
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -L/opt/rocm/lib -lm")
endif()

if (${USE_OPENCL})
    install(DIRECTORY lib/opencl/kernels DESTINATION /var/lib/kotekan/opencl )
endif ()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Required for some SSE/AVX intrinsics
#add_definitions(-mssse3 -D__STDC_LIMIT_MACROS)


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -march=native -mtune=native -I/opt/rocm/include")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb ")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Note C++11
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -Wall -march=native -mtune=native -I/opt/rocm/include")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb")
#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")
#SET(CMAKE_INSTALL_PREFIX "/")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (${COMPILE_DOCS})
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(include)
