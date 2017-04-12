#ifndef NETWORK_POWER_STREAM_H
#define NETWORK_POWER_STREAM_H

#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"

class networkPowerStream : public KotekanProcess {
public:
    networkPowerStream(Config &config, struct Buffer &buf);
    virtual ~networkPowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer &buf;

    uint32_t dest_port;
    string dest_server_ip;


    int freqs;
    int times;
};

#endif