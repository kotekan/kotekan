#ifndef NETWORK_DPDK
#define NETWORK_DPDK

#include "buffers.h"
#include "errors.h"

#define NUM_LINKS (4)

struct networkDPDKArg {
    // Array of output buffers
    struct Buffer * buf;
    int num_links;

    struct Config * config;
};

struct LinkData {
    int64_t seq;
    int64_t last_seq;
    int64_t seq64;
    int64_t cur_seq64_edge;
    uint16_t stream_ID;
    int64_t lost_packets;
    uint32_t num_packets;
    int32_t buffer_id;
    int32_t finished_buffer;
};

struct NetworkDPDK {

    struct LinkData link_data[NUM_LINKS];

    double start_time;
    double end_time;

    uint32_t data_id;
    uint32_t num_unused_cycles;

    struct networkDPDKArg * args;
};

void network_dpdk_thread(void * arg);

#endif