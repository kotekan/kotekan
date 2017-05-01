# Inspect Packet section
#add_executable(inspect_pkt inspect_pkt.c)
#target_link_libraries(inspect_pkt pfring.a)

# Data dump from kotekan
add_executable(gpu_stream_dump gpu_stream_dump.c)

# Include chime lib
include_directories (${CHIME_SOURCE_DIR}/lib)
link_directories (${CHIME_BINARY_DIR}/lib)
include_directories( ${OPENCL_INCLUDE_DIRS} )

# Remove PF_RING for now.
#add_executable(adc_dump adc_dump.c)
#target_link_libraries(adc_dump chime)
#target_link_libraries(adc_dump pfring.a)

target_link_libraries(gpu_stream_dump hdf5)
target_link_libraries(gpu_stream_dump hdf5_hl)