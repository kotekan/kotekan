cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)

add_library ( kotekan_libs
              KotekanProcess.cpp
              SampleProcess.cpp
              Config.cpp
              buffers.c
              errors.c
              nt_memcpy.c
              nt_memset.c
              restServer.cpp
              util.c
              error_correction.c
              output_formating.c
              time_tracking.c
              fpga_header_functions.c
              network_dpdk.c
              bufferContainer.cpp
              processFactory.cpp
              )

#set (DPDK_LIBS "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_pmd_i40e -lrte_mempool -lrte_ring -lethdev -lrte_pmd_ixgbe -lrte_malloc -Wl,--no-whole-archive" )

target_link_libraries( kotekan_libs ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( kotekan_libs m rt dl ${DPDK_LIBS} )
target_link_libraries( kotekan_libs libinclude )

#if (${USE_DPDK})
    include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
    link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
#endif ()

if (${USE_HCC})
       add_subdirectory(hcc)
       target_link_libraries( kotekan_libs kotekan_hcc )
endif ()

if (${USE_HSA})
       add_subdirectory(hsa)
       target_link_libraries( kotekan_libs kotekan_hsa )
       target_link_libraries( kotekan_libs hsa-runtime64 )
endif ()

if (${USE_OPENCL})
#       add_subdirectory(opencl)
#       target_link_libraries( kotekan_libs kotekan_opencl )
       add_subdirectory(opencl)
       target_link_libraries( kotekan_libs ${OPENCL_LIBRARIES} )
       target_link_libraries( kotekan_libs kotekan_opencl )
endif ()

add_subdirectory(testing)
target_link_libraries( kotekan_libs kotekan_testing )
add_subdirectory(processes)
target_link_libraries( kotekan_libs kotekan_processes )

