cmake_minimum_required( VERSION 2.8 )
project( libchime )

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories (${CHIME_SOURCE_DIR}/include)

include_directories (${CHIME_SOURCE_DIR}/lib)
include_directories (${CHIME_SOURCE_DIR}/lib/gpu_command_objects)

add_library ( chime
              KotekanProcess.cpp SampleProcess.cpp Config.cpp
              buffers.c errors.c nt_memcpy.c nt_memset.c
              util.c error_correction.c networkOutputSim.cpp
              output_formating.c test_data_generation.c chrxUplink.cpp gpuPostProcess.cpp
              beamformingPostProcess.cpp time_tracking.c vdifStream.cpp
              fpga_header_functions.c nullProcess.cpp simple_udp_cap.c output_power.c
              network_dpdk.c gpu_thread.cpp gpu_command.cpp callbackdata.cpp device_interface.cpp
              gpu_command_factory.cpp vdif_functions.c stream_raw_vdif.c
              gpu_command_objects/beamform_kernel.cpp
              gpu_command_objects/beamform_phase_data.cpp
              gpu_command_objects/correlator_kernel.cpp
	          gpu_command_objects/input_data_stage.cpp
              gpu_command_objects/offset_kernel.cpp
	          gpu_command_objects/output_beamform_result.cpp
              gpu_command_objects/output_data_result.cpp
              gpu_command_objects/preseed_kernel.cpp
              gpu_command_objects/beamform_incoherent_kernel.cpp
              gpu_command_objects/output_beamform_incoh_result.cpp
              )

#set (DPDK_LIBS "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_pmd_i40e -lrte_mempool -lrte_ring -lethdev -lrte_pmd_ixgbe -lrte_malloc -Wl,--no-whole-archive" )

target_link_libraries( chime ${OPENCL_LIBRARIES} )
target_link_libraries( chime ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( chime m rt dl ${DPDK_LIBS} )
