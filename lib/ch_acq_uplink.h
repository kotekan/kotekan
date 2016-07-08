#ifndef CH_ACQ_UPLINK
#define CH_ACQ_UPLINK

#ifdef __cplusplus
extern "C" {
#endif

struct ch_acqUplinkThreadArg {
    struct Config * config;
    struct Buffer * buf;
    struct Buffer * gate_buf;
};

void* ch_acq_uplink_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif