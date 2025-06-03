#ifndef VDIF_STREAM
#define VDIF_STREAM

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string

class vdifStream : public kotekan::Stage {
public:
    vdifStream(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~vdifStream();
    void main_thread() override;

private:
    Buffer* buf;

    uint32_t _vdif_port;
    std::string _vdif_server_ip;
};

#endif
