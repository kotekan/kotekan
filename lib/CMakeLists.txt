cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)

set (KOTEKAN_LIB_SOURCES
              KotekanProcess.cpp
              SampleProcess.cpp
              Config.cpp
              buffer.c
              chimeMetadata.c
              metadata.c
              errors.c
              nt_memcpy.c
              nt_memset.c
              restServer.cpp
              util.c
              output_formating.c
              time_tracking.c
              fpga_header_functions.c
              bufferContainer.cpp
              processFactory.cpp
              metadataFactory.cpp
              bufferFactory.cpp
              configEval.cpp
              visUtil.cpp
       )

if (${USE_DPDK})
       set (KOTEKAN_LIB_SOURCES ${KOTEKAN_LIB_SOURCES} network_dpdk.c)
endif ()

if (${USE_HDF5})
  set (KOTEKAN_LIB_SOURCES ${KOTEKAN_LIB_SOURCES} visFile.cpp)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (KOTEKAN_LIB_SOURCES ${KOTEKAN_LIB_SOURCES} osxBindCPU.cpp)
#else ()
#       set (KOTEKAN_LIB_SOURCES ${KOTEKAN_LIB_SOURCES} simple_udp_cap.c)
endif ()

add_library ( kotekan_libs
              ${KOTEKAN_LIB_SOURCES})

target_link_libraries( kotekan_libs ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( kotekan_libs libinclude )
target_link_libraries( kotekan_libs m )

if (${USE_DPDK})
    include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
    link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
endif ()

if (${USE_HCC})
       add_subdirectory(hcc)
       target_link_libraries( kotekan_libs kotekan_hcc )
endif ()

if (${USE_HSA})
       add_subdirectory(hsa)
       link_directories ( /opt/rocm/hsa/lib/ )
       target_link_libraries( kotekan_libs kotekan_hsa )
       target_link_libraries( kotekan_libs hsa-runtime64 )
       include_directories ( /opt/rocm/hsa/include/ )
endif ()

if (${USE_OPENCL})
       add_subdirectory(opencl)
       target_link_libraries( kotekan_libs ${OPENCL_LIBRARIES} )
       target_link_libraries( kotekan_libs kotekan_opencl )
endif ()

# HDF5 stuff
if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  include_directories (/kotekan/HighFive/include/)
  target_link_libraries ( kotekan_libs ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
endif ()

add_subdirectory(testing)
target_link_libraries( kotekan_libs kotekan_testing )
add_subdirectory(processes)
target_link_libraries( kotekan_libs kotekan_processes )
