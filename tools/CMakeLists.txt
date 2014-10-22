# Inspect Packet section
add_executable(inspect_pkt inspect_pkt.c)

target_link_libraries(inspect_pkt  pfring.a )
