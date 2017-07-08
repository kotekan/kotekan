#ifndef BUFFER
#define BUFFER

#ifdef __cplusplus
extern "C" {
#endif

struct chimeMetadata {
    int64_t fpga_seq_num;
    struct timeval first_packet_recv_time;
    int32_t lost_timesamples;
    uint16_t stream_ID;
};

#ifdef __cplusplus
}
#endif

#endif