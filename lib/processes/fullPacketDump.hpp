#ifndef FULL_PACKET_DUMP_HPP
#define FULL_PACKET_DUMP_HPP

#include "KotekanProcess.hpp"
#include "restServer.hpp"
#include <string>
#include <mutex>

class fullPacketDump : public KotekanProcess {
public:
    fullPacketDump(Config& config, const string& unique_name,
                   bufferContainer &buffer_container);
    virtual ~fullPacketDump();
    void main_thread() override;

    void packet_grab_callback(connectionInstance& conn, json& json_request);

private:
    struct Buffer *buf;
    int link_id;

    bool got_packets = false;

    int _packet_size;
    uint8_t * _packet_frame;
    bool _dump_to_disk = true;
    std::string _file_base;
    std::string _data_set;
    std::mutex _packet_frame_lock;
    std::string endpoint;
};

#endif