#ifndef NETWORK_OUTPUT_SIM
#define NETWORK_OUTPUT_SIM

#define SIM_CONSTANT   0
#define SIM_FULL_RANGE 1
#define SIM_SINE        2

#include "buffers.h"
#include "errors.h"

#ifdef __cplusplus
extern "C" {
#endif

struct networkOutputSim {

    struct Buffer * buf;
    int num_links_in_group;
    int link_id;
    int pattern;
    int stream_id;

    struct Config * config;
};

void* network_output_sim(void * arg);


#ifdef __cplusplus
}
#endif

#endif