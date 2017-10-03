#ifndef STREAM_SINGLE_DISH_VDIF_H
#define STREAM_SINGLE_DISH_VDIF_H

#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"

class streamSingleDishVDIF : public KotekanProcess {
public:
    streamSingleDishVDIF(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);
    virtual ~streamSingleDishVDIF();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer *buf;

    uint32_t vdif_dest_port;
    string vdif_dest_ip;
    int num_freq;

};

#endif
