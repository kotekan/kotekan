cmake_minimum_required (VERSION 2.8)
project (CHIME)
set(CMAKE_MODULE_PATH ${CHIME_SOURCE_DIR}/cmake)

option(USE_HCC "Build HCC GPU Framework" OFF)
option(USE_HSA "Build HSA GPU Framework" ON)
option(USE_OPENCL "Build OpenCL GPU Framework" OFF)

if (${USE_HCC})
  find_package( HCC )
  set(USE_HCC ${HCC_fOUND})
endif ()
if (${USE_HSA})
  find_package( HSA )
  set(USE_HSA ${HSA_FOUND})
endif ()
if (${USE_OPENCL})
  find_package( OpenCL )
  set(USE_OPENCL ${OPENCL_FOUND})
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

# Include external software.
include_directories (${CHIME_SOURCE_DIR}/include)
if (${USE_HCC})
  include_directories (${CHIME_SOURCE_DIR}/lib/hcc)
endif ()

if (${USE_HSA})
  include_directories (${CHIME_SOURCE_DIR}/lib/hsa)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lhsa-runtime64 ")
  add_definitions(-DWITH_HSA)
endif ()

if (${USE_OPENCL})
  include_directories (${CHIME_SOURCE_DIR}/lib/opencl)
  include_directories (${CHIME_SOURCE_DIR}/lib/old_opencl)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lcl ")
  add_definitions(-DWITH_OPENCL)
endif ()

#include_directories (${CHIME_SOURCE_DIR}/lib/sim_output)

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

# Required for some SSE/AVX intrinsics
add_definitions(-mssse3 -D__STDC_LIMIT_MACROS)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -march=native -mtune=native -I/opt/rocm/include")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -ggdb ")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Note C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -Wall -march=native -mtune=native -I/opt/rocm/include")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -ggdb")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

SET(CMAKE_INSTALL_PREFIX "/")

install(DIRECTORY kernels DESTINATION /var/lib/kotekan/ )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++ -L/opt/rocm/lib -lm")

#/// AMD hcc options

if (CMAKE_CXX_COMPILER MATCHES ".*hcc")
    # set the compile options for hcc
    execute_process(COMMAND hcc-config --cxxflags OUTPUT_VARIABLE HCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_COMPILE_FLAGS} " )

    # set the link options for hcc
    execute_process(COMMAND hcc-config --ldflags OUTPUT_VARIABLE HCC_LINKER_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HCC_LINKER_FLAGS} ")

    add_definitions(-DWITH_HCC)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

#/// end AMD hcc options

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(include)
#add_subdirectory(vdif_dump)
