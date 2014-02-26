#ifndef NETWORK
#define NETWORK

#include "buffers.h"

struct network_thread_arg {
    char * ip_address;
    struct Buffer * buf;
    int portNumber;
    int bufferDepth;
    int data_limit;
    int numLinks;
    int link_id;
};

void network_thread(void * arg);

#endif