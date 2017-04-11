cmake_minimum_required( VERSION 2.8 )
project( libchime )

include_directories (${CHIME_SOURCE_DIR}/include)

include_directories (${CHIME_SOURCE_DIR}/lib)

add_library ( chime
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
              networkOutputSim.cpp
              output_formating.c
              test_data_generation.c
              chrxUplink.cpp
              gpuPostProcess.cpp
              beamformingPostProcess.cpp
              time_tracking.c
              vdifStream.cpp
              fpga_header_functions.c
              nullProcess.cpp
              simple_udp_cap.c
              output_power.c
              network_dpdk.c
              fullPacketDump.cpp
              dpdkWrapper.cpp
              rawFileRead.cpp
              rawFileWrite.cpp
              pyPlotResult.cpp 
              bufferContainer.cpp
              testDataCheck.cpp
              testDataGen.cpp
              computeDualpolPower.cpp
              )

#set (DPDK_LIBS "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_pmd_i40e -lrte_mempool -lrte_ring -lethdev -lrte_pmd_ixgbe -lrte_malloc -Wl,--no-whole-archive" )

target_link_libraries( chime ${OPENCL_LIBRARIES} )
target_link_libraries( chime ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( chime m rt dl ${DPDK_LIBS} )
target_link_libraries( chime libinclude )

if (${USE_HCC})
       add_subdirectory(hcc)
       target_link_libraries( chime kotekan_hcc )
endif ()

if (${USE_HSA})
       add_subdirectory(hsa)
       target_link_libraries( chime kotekan_hsa )
endif ()

if (${USE_OPENCL})
#       add_subdirectory(opencl)
#       target_link_libraries( chime kotekan_opencl )
       add_subdirectory(old_opencl)
       target_link_libraries( chime kotekan_old_opencl )
endif ()

add_subdirectory(sim_output)
target_link_libraries( chime kotekan_sim_output )


