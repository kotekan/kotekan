#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "nullProcess.hpp"
#include "buffers.h"
#include "errors.h"
#include "output_formating.h"
#include "fullPacketDump.hpp"
#include "restServer.hpp"

#define MAX_NUM_PACKETS 100

fullPacketDump::fullPacketDump(Config& config, struct Buffer &buf_, int link_id_) :
    KotekanProcess(config, std::bind(&fullPacketDump::main_thread, this)),
    buf(buf_), link_id(link_id_){

    apply_config(0);
    _packet_frame = (uint8_t*)malloc(_packet_size * MAX_NUM_PACKETS);
}

fullPacketDump::~fullPacketDump() {
    free(_packet_frame);
}

void fullPacketDump::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    _packet_size = config.get_int("/fpga_network/udp_packet_size");
}

uint8_t* fullPacketDump::packet_grab_callback(int num_packets, int& len) {
    assert(num_packets <= MAX_NUM_PACKETS);

    if (!got_packets)
        return nullptr;

    len = num_packets * _packet_size;

    return _packet_frame;
}

void fullPacketDump::main_thread() {
    int buffer_ID = 0;

    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    rest_server->register_packet_callback(std::bind(&fullPacketDump::packet_grab_callback, this, _1, _2), link_id);

    // Wait for, and drop full buffers
    while (!stop_thread) {

        // This call is blocking!
        buffer_ID = get_full_buffer_from_list(&buf, &buffer_ID, 1);
        //INFO("fullPacketDump: link %d got full full buffer ID %d", link_id, buffer_ID);
        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            break;
        }

        memcpy(_packet_frame, buf.data[buffer_ID], _packet_size * MAX_NUM_PACKETS);
        if (!got_packets) got_packets = true;

        release_info_object(&buf, buffer_ID);
        mark_buffer_empty(&buf, buffer_ID);

        buffer_ID = (buffer_ID + 1) % buf.num_buffers;
    }
    INFO("Closing full packet dump thread");
}
