#ifndef STREAM_SINGLE_DISH_VDIF_H
#define STREAM_SINGLE_DISH_VDIF_H

#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"

class streamSingleDishVDIF : public KotekanProcess {
public:
    streamSingleDishVDIF(Config &config, struct Buffer &buf);
    virtual ~streamSingleDishVDIF();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer &buf;

    uint32_t _vdif_port;
    string _vdif_ip;
    int _num_freq;

};

#endif

