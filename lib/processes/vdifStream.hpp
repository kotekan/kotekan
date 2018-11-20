#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"

class vdifStream : public KotekanProcess {
public:
    vdifStream(Config& config, const string& unique_name,
               bufferContainer &buffer_container);
    virtual ~vdifStream();
    void main_thread() override;

private:
    struct Buffer *buf;

    uint32_t _vdif_port;
    string _vdif_server_ip;

};

#endif