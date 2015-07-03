cmake_minimum_required( VERSION 2.6 )
project( libchime )

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

add_library ( chime buffers.c errors.c nt_memcpy.c gpu_thread.c network_dna.c util.c error_correction.c output_formating.c test_data_generation.c ch_acq_uplink.c gpu_post_process.c config.c beamforming.c beamforming_post_process.c time_tracking.c vdif_stream.c fpga_header_functions.c null.c)

target_link_libraries( chime ${OPENCL_LIBRARIES} )
target_link_libraries( chime ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( chime m )
target_link_libraries( chime rt )
target_link_libraries( chime pfring )
target_link_libraries( chime jansson )