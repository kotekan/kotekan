#ifndef ACQ_UPLINK_H
#define ACQ_UPLINK_H

#include <stdint.h>             // for int32_t
#include <string>               // for string

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "bufferContainer.hpp"  // for bufferContainer
#include "buffer.hpp"           // for Buffer


class chrxUplink : public kotekan::Stage {
public:
    chrxUplink(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~chrxUplink();
    void main_thread() override;

private:
    Buffer* vis_buf;
    Buffer* gate_buf;

    // Config variables
    std::string _collection_server_ip;
    int32_t _collection_server_port;
    bool _enable_gating;
};

#endif
