#ifndef ACQ_UPLINK_H
#define ACQ_UPLINK_H

#include "Stage.hpp" // for Stage

#include <stdint.h> // for int32_t
#include <string>   // for string

namespace kotekan {
class Config;
class bufferContainer;
} // namespace kotekan


class chrxUplink : public kotekan::Stage {
public:
    chrxUplink(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    virtual ~chrxUplink();
    void main_thread() override;

private:
    struct Buffer* vis_buf;
    struct Buffer* gate_buf;

    // Config variables
    std::string _collection_server_ip;
    int32_t _collection_server_port;
    bool _enable_gating;
};

#endif
