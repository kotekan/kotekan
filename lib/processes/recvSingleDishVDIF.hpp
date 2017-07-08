#ifndef RECV_SINGLE_DISH_VDIF_H
#define RECV_SINGLE_DISH_VDIF_H

#include "buffer.c"
#include "KotekanProcess.hpp"
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

class recvSingleDishVDIF : public KotekanProcess {
public:
    recvSingleDishVDIF(Config& config, const string& unique_name,
                bufferContainer &buffer_container);
    virtual ~recvSingleDishVDIF();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;

    uint32_t vdif_orig_port;
    string vdif_orig_ip;
    int num_freq;
};

#endif