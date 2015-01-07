#ifndef GPU_POST_PROCESS
#define GPU_POST_PROCESS

struct gpuPostProcessThreadArg {
    struct Config * config;
    struct Buffer * in_buf;
    struct Buffer * out_buf;
};

#define MAX_NUM_LINKS (8)

// A TCP frame contains this header followed by the visibilities, and flags.
// -- HEADER:sizeof(TCP_frame_header) --
// -- VISIBILITIES:n_corr * n_freq * sizeof(complex_int_t) --
// -- FLAGS:n_corr * sizeof(uint8_t) --
#pragma pack(1)
struct stream_id {
    unsigned int link_id : 8;
    unsigned int slot_id : 8;
    unsigned int crate_id : 8;
    unsigned int reserved : 8;
};

struct tcp_frame_header {
    uint32_t fpga_seq_number;
    uint32_t num_freq;
    uint32_t num_vis; // The number of visibilities per frequency.

    struct stream_id stream_ids[MAX_NUM_LINKS];

    struct timeval cpu_timestamp; // The time stamp as set by the GPU correlator - not accurate!
};
#pragma pack(0)

void gpu_post_process_thread(void * arg);

#endif