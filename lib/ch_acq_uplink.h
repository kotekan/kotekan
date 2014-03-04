#ifndef CH_ACQ_UPLINK
#define CH_ACQ_UPLINK

struct ch_acqUplinkThreadArg {
    struct Buffer * buf;
    int num_links;
    int buffer_depth;
    char * ch_acq_ip_addr;
    int ch_acq_port_num;

    int actual_num_freq;
    int actual_num_elements;
};


void ch_acq_uplink_thread(void * arg);

#endif