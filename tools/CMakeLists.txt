# Inspect Packet section
add_executable(inspect_pkt inspect_pkt.c)

target_link_libraries(inspect_pkt  pfring.a )

# Inspect Packet Server section
add_executable(inspect_pkt_server inspect_pkt_server.c)
target_link_libraries(inspect_pkt_server  pfring.a )

