#ifndef NETWORK_DPDK
#define NETWORK_DPDK

#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"

// TODO Make these dynamic.
#ifdef DPDK_VDIF_MODE
    #define NUM_LINKS (8)
    #define NUM_LCORES (4)
    // This shouldn't go above 4, since it's for the shuffle.
    // A better name might be SHUFFLE_SIZE?
    #define NUM_FREQ (1)
    #define MAX_CORES (8)
#else
    #ifdef WITH_OPENCL
        // Pathfinder mode
        #define NUM_LINKS (4)
        #define NUM_FREQ (1)
        #define MAX_CORES (8)
    #else
        // CHIME Mode
        #define NUM_LINKS (4)
        #define NUM_FREQ (4)
        #define MAX_CORES (12)
    #endif

    #define NUM_LCORES (4)
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct networkDPDKArg {
    // Array of output buffers
    // TODO Not sure I like a triple pointer.
    // *[port][freq]
    struct Buffer *** buf;

    // These should take over the defines.
    int num_links;
    int num_lcores;
    int num_links_per_lcore;

    uint32_t num_links_in_group[NUM_LINKS];
    uint32_t link_id[NUM_LINKS];
    uint32_t port_offset[NUM_LCORES];
    /// Maps lcores to ports.  Note lcore == -1 always mapps to port 0.
    uint32_t lcore_port_mapping[MAX_CORES];

    int32_t timesamples_per_packet;
    int32_t samples_per_data_set;
    int32_t num_data_sets;
    int32_t num_gpu_frames;
    int32_t udp_packet_size;

    int dump_full_packets;
    int enable_shuffle;

    // Note ideally this would hvae atomic access,
    // but since only one tread writes and others
    // read it should be safe and much more efficent
    // to have it not be atomic.
    int stop_capture;

    // Used for the vdif generation
    struct Buffer * vdif_buf;

    // Fake stream ids
    int fake_stream_ids;

    // The producer names
    char producer_names[NUM_LINKS][MAX_PROCESS_NAME_LEN];

    // ** Shared information with wrapper **
    uint64_t rx_packets_total[NUM_LINKS];
    uint64_t rx_bytes_total[NUM_LINKS];
    uint64_t lost_packets_total[NUM_LINKS];
    uint64_t rx_errors_total[NUM_LINKS];
};

struct LinkData {
    uint64_t seq;
    uint64_t last_seq;
    uint16_t stream_ID; // TODO just use the struct for this.
    stream_id_t s_stream_ID;
    int32_t first_packet;
    int32_t buffer_id;
    int32_t vdif_buffer_id;
    int32_t finished_buffer;
    int32_t data_id;
    int32_t dump_location;
};

struct NetworkDPDK {

    struct LinkData link_data[NUM_LINKS][NUM_FREQ];

    double start_time;
    double end_time;

    uint32_t data_id;
    uint32_t num_unused_cycles;

    struct networkDPDKArg * args;

    int vdif_time_set;
    uint64_t vdif_offset;  // Take (seq - offset) mod 5^8 to get data frame
    uint64_t vdif_base_time; // Add this to (seq - offset) / 5^8 to get time.
};

void* network_dpdk_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif
