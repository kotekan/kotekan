#ifndef FULL_PACKET_DUMP_HPP
#define FULL_PACKET_DUMP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <mutex>    // for mutex
#include <stdint.h> // for uint8_t
#include <string>   // for string, basic_string

/**
 * @class fullPacketDump
 * @brief Captures raw UDP packets from a network buffer and either writes them to disk or exposes
 * them over REST.
 *
 * Reads frames from `network_in_buf` (typically raw packet captures). If `dump_to_disk` is true it
 * writes each frame to `file_base/data_set/<host>_<link_id>_<seq>.pkt`. When `dump_to_disk` is
 * false, the most recent frame (up to 100 packets) is cached and retrievable via
 * POST `/unique_name/packet_grab/<link_id>` with JSON `{"num_packets": N}`; the endpoint returns
 * the first N packets of the cached frame. No output buffers are produced.
 *
 * @par Buffers
 * @buffer network_in_buf Raw network packet buffer.
 *     @buffer_format uint8_t payload (UDP packets concatenated)
 *     @buffer_metadata none
 *
 * @conf link_id         Int. Logical link index (used in REST path and filename).
 * @conf udp_packet_size Int. Size of each UDP packet in bytes.
 * @conf dump_to_disk    Bool. If true write frames to disk; otherwise cache for REST.
 * @conf file_base       String. Base directory for output files (if dumping).
 * @conf data_set        String. Subdirectory name under file_base (if dumping).
 *
 * @par REST
 * @restendpoint POST /<unique_name>/packet_grab/<link_id> Body: `{"num_packets": N}`; returns
 *              N packets from the latest frame when `dump_to_disk` is false.
 *
 * @par Example
 * @code
 * full_packet_dump:
 *   kotekan_stage: fullPacketDump
 *   network_in_buf: raw_net
 *   link_id: 0
 *   udp_packet_size: 4928
 *   dump_to_disk: true
 *   file_base: /data/packets
 *   data_set: test_run
 * @endcode
 */
class fullPacketDump : public kotekan::Stage {
public:
    fullPacketDump(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    virtual ~fullPacketDump();
    void main_thread() override;

    void packet_grab_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

private:
    Buffer* buf;
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
