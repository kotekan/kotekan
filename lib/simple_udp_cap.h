#ifndef SIMPLE_UDP_CAP
#define SIMPLE_UDP_CAP

#include "buffers.h"

struct udpCapArgs {
    char * ip_address;
    struct Buffer * buf;
    int port_number;
    int buffer_depth;
    int data_limit;
    int num_links;
    int link_id;
};

void simple_udp_cap(void * arg);

#endif