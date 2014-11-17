# Inspect Packet section
add_executable(inspect_pkt inspect_pkt.c)

target_link_libraries(inspect_pkt  pfring.a )

# Data dump from kotekan
add_executable(gpu_stream_dump gpu_stream_dump.c)

target_link_libraries( gpu_stream_dump hdf5 )
target_link_libraries( gpu_stream_dump hdf5_hl )