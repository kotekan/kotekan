cmake_minimum_required (VERSION 2.8)
project (KOTEKAN)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

#optional modules for GPUs and the like
option(USE_HCC "Build HCC GPU Framework" OFF)
option(USE_HSA "Build HSA GPU Framework" OFF)
option(USE_OPENCL "Build OpenCL GPU Framework" OFF)
option(USE_DPDK OFF)

if (${USE_HCC} AND (CMAKE_CXX_COMPILER MATCHES ".*hcc"))
  find_package( HCC )
  set(USE_HCC ${HCC_fOUND})
else ()
  set(USE_HCC FALSE)
endif ()
if (${USE_HSA})
  find_package( HSA )
  set(USE_HSA ${HSA_FOUND})
endif ()
if (${USE_OPENCL})
  find_package( OpenCL )
  set(USE_OPENCL ${OPENCL_FOUND})
endif ()


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_definitions(-DMAC_OSX)
  add_definitions(-mavx2)
else()
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()


set(MODULES "")
if (${USE_HCC})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/hcc)
  add_definitions(-DWITH_HCC)
  set(MODULES ${MODULES} "HCC ")
endif ()
if (${USE_HSA})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lhsa-runtime64 ")
  add_definitions(-DWITH_HSA)
  set(MODULES ${MODULES} "HSA ")
endif ()
if (${USE_OPENCL})
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
  include_directories (${KOTEKAN_SOURCE_DIR}/lib/old_opencl)
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL ")
  add_definitions(-DWITH_OPENCL)
  set(MODULES ${MODULES} "OpenCL")
endif ()
MESSAGE("GPU Modules Included: " ${MODULES})

if (${USE_DPDK})
  # DPDK Includes
  include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
  link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
  
  set (DPDK_LIST
          ethdev       rte_distributor  rte_kni     rte_mbuf      rte_pmd_bond   rte_pmd_ring
          rte_power    rte_timer        rte_ring    rte_sched     rte_table
          rte_acl      rte_eal          rte_kvargs  rte_mempool   rte_pmd_e1000
          rte_cfgfile  rte_hash         rte_lpm     rte_meter     rte_pmd_i40e   rte_pmd_vmxnet3_uio
          rte_cmdline  rte_ip_frag      rte_pipeline  rte_pmd_ixgbe  rte_port
  )
  set (DPDK_LIBS
        -Wl,--start-group ${DPDK_LIST} -Wl,--end-group )

  add_definitions(-DWITH_DPDK)
  MESSAGE("DPDK Enabled")
else ()
  MESSAGE("DPDK Disabled")
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


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -march=native -mtune=native")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb ")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Note C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -Wall -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")
SET(CMAKE_INSTALL_PREFIX "/")
install(DIRECTORY kernels DESTINATION /var/lib/kotekan/ )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(include)
#add_subdirectory(vdif_dump)
