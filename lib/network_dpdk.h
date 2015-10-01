#ifndef NETWORK_DPDK
#define NETWORK_DPDK

#include "buffers.h"
#include "errors.h"

struct networkDPDKArg {
    struct Buffer * buf;
    int num_links;

    struct Config * config;
};

void network_dpdk_thread(void * arg);

#endif