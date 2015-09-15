# Inspect Packet section
add_executable(inspect_pkt inspect_pkt.c)

target_link_libraries(inspect_pkt  pfring.a )

# Data dump from kotekan
add_executable(gpu_stream_dump gpu_stream_dump.c)

# Code to recv data from an aro ram disk ring buffer from another system
add_executable(ring_recv ring_recv.c)

# Code to send data to an aro ram disk ring buffer to another system
add_executable(ring_send ring_send.c)

target_link_libraries( gpu_stream_dump hdf5 )
target_link_libraries( gpu_stream_dump hdf5_hl )