#ifndef NETWORK
#define NETWORK

#include "buffers.h"
#include "errors.h"

struct networkThreadArg {
    char * ip_address;
    struct Buffer * buf;
    int port_number;
    int buffer_depth;
    int data_limit;
    int num_links;
    int link_id;

    // Args used for testing.
    int num_timesamples;
    int actual_num_freq;
    int actual_num_elements;

    int read_from_file;
    char * file_name;
};

void network_thread(void * arg);

void test_network_thread(void * arg);

#endif