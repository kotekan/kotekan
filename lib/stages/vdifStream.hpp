#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"

class vdifStream : public kotekan::Stage {
public:
    vdifStream(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~vdifStream();
    void main_thread() override;

private:
    struct Buffer* buf;

    uint32_t _vdif_port;
    std::string _vdif_server_ip;
};

#endif