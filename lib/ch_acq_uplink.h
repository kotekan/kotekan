#ifndef CH_ACQ_UPLINK
#define CH_ACQ_UPLINK

#include "link_mapping.h"

struct ch_acqUplinkThreadArg {
    struct Config * config;
    struct Buffer * buf;
};


void ch_acq_uplink_thread(void * arg);

#endif