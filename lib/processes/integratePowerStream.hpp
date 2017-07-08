#ifndef INTEGRATE_POWER_STREAM_H
#define INTEGRATE_POWER_STREAM_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.c"
#include "KotekanProcess.hpp"
#include <atomic>

class integratePowerStream : public KotekanProcess {
public:
    integratePowerStream(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container);
    virtual ~integratePowerStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
	void tcpConnect();

    struct Buffer *buf_in;
    struct Buffer *buf_out;

    int freqs;
    int times;
    int elems;

    int integration_length;
};

#endif