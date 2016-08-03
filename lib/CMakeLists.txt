cmake_minimum_required( VERSION 2.8 )
project( libchime )

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

add_library ( chime buffers.c errors.c nt_memcpy.c nt_memset.c raw_file_write.c
              util.c error_correction.c network_output_sim.c
              output_formating.c test_data_generation.c ch_acq_uplink.c gpu_post_process.c
              config.c beamforming.c beamforming_post_process.c time_tracking.c vdif_stream.c
              fpga_header_functions.c null.c file_write.c simple_udp_cap.c output_power.c
              network_dpdk.c gpu_thread.cpp gpu_command.cpp callbackdata.cpp device_interface.cpp
              gpu_command_factory.cpp vdif_functions.c stream_raw_vdif.c beamform_data_stage.cpp
	      beamform_kernel.cpp beamform_phase_data.cpp correlator_kernel.cpp
	      dummy_placeholder_kernel.cpp input_data_stage.cpp offset_kernel.cpp
	      output_beamform_result.cpp output_data_result.cpp preseed_kernel.cpp)

#set (DPDK_LIBS "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_pmd_i40e -lrte_mempool -lrte_ring -lethdev -lrte_pmd_ixgbe -lrte_malloc -Wl,--no-whole-archive" )

target_link_libraries( chime ${OPENCL_LIBRARIES} )
target_link_libraries( chime ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( chime m rt dl ${DPDK_LIBS} )
target_link_libraries( chime jansson )
