#ifndef NETWORK_DPDK
#define NETWORK_DPDK

#include "buffers.h"
#include "errors.h"

// TODO Make these dynamic.
#define NUM_LINKS (8)
#define NUM_LCORES (4)

#ifdef __cplusplus
extern "C" {
#endif

struct networkDPDKArg {
    // Array of output buffers
    struct Buffer ** buf;

    // These should take over the defines.
    int num_links;
    int num_lcores;
    int num_links_per_lcore;

    struct Config * config;
    uint32_t integration_edge_offset;
    uint32_t num_links_in_group[NUM_LINKS];
    uint32_t link_id[NUM_LINKS];
    uint32_t port_offset[NUM_LCORES];
};

struct LinkData {
    int64_t seq;
    int64_t last_seq;
    int64_t seq64;
    int64_t last_seq64;
    int64_t cur_seq64_edge;
    uint16_t stream_ID;
    int64_t lost_packets;
    uint32_t num_packets;
    int32_t buffer_id;
    int32_t finished_buffer;
    int32_t data_id;
};

struct NetworkDPDK {

    struct LinkData link_data[NUM_LINKS];

    double start_time;
    double end_time;

    uint32_t data_id;
    uint32_t num_unused_cycles;

    struct networkDPDKArg * args;
};

void* network_dpdk_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif
