#ifndef FULL_PACKET_DUMP_HPP
#define FULL_PACKET_DUMP_HPP

#include "Config.hpp"          // IWYU pragma: keep
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // IWYU pragma: keep

#include "json.hpp" // for json

#include <mutex>    // for mutex
#include <stdint.h> // for uint8_t
#include <string>   // for string

namespace kotekan {
class connectionInstance;
} // namespace kotekan

class fullPacketDump : public kotekan::Stage {
public:
    fullPacketDump(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    virtual ~fullPacketDump();
    void main_thread() override;

    void packet_grab_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

private:
    struct Buffer* buf;
    int link_id;

    bool got_packets = false;

    int _packet_size;
    uint8_t* _packet_frame;
    bool _dump_to_disk = true;
    std::string _file_base;
    std::string _data_set;
    std::mutex _packet_frame_lock;
    std::string endpoint;
};

#endif
