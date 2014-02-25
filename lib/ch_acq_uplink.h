
#ifndef CH_ACQ_UPLINK
#define CH_ACQ_UPLINK

struct ch_acq_uplink_thread_arg {
    struct Buffer * buf;
    int num_links;
    int bufferDepth;
    char * ch_acq_ip_addr;
    int ch_acq_port_num;

    int num_elem;
    int num_freq;
};


void ch_acq_uplink_thread(void * arg);

#endif