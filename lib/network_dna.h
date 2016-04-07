#ifndef NETWORK
#define NETWORK

#include "buffers.h"
#include "errors.h"

#ifdef __cplusplus
extern "C" {
#endif

struct networkThreadArg {
    char * ip_address;
    struct Buffer * buf;
    int data_limit;
    int num_links_in_group;
    int link_id;
    int dev_id;

    struct Config * config;

    int read_from_file;
    char * file_name;
};

void* network_thread(void * arg);

void* test_network_thread(void * arg);

#ifdef __cplusplus
}
#endif

#endif