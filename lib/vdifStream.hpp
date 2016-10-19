#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "Config.hpp"
#include "buffers.h"
#include "KotekanProcess.hpp"

class vdifStream : public KotekanProcess {
public:
    vdifStream(struct Config &config, struct Buffer &buf);
    virtual ~vdifStream();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer &buf;

    uint32_t _vdif_port;
    string _vdif_server_ip;

};

#endif