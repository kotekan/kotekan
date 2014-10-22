#ifndef NETWORK
#define NETWORK

#include "buffers.h"

struct network_thread_arg {
    char * interface;
    struct Buffer * buf;
    int portNumber;
    int bufferDepth;
    int data_limit;
    int numLinks;
    int link_id;

    int num_frames;
    int num_inputs;
    int num_freq;
    int offset;
};

void network_thread(void * arg);

#endif