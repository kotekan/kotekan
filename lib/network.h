#ifndef NETWORK
#define NETWORK

#include "buffers.h"

struct networkThreadArg {
    char * ip_address;
    struct Buffer * buf;
    int port_number;
    int buffer_depth;
    int data_limit;
    int num_links;
    int link_id;
};

void network_thread(void * arg);

#endif