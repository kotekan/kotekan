#include "udpPacketGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO
#include "udpPacket.hpp"       // for fill_udp_headers, parse_mac, parse_ipv4, udp_app_header
#include "visUtil.hpp"         // for frameID

#include "fmt.hpp" // for format

#include <functional> // for bind
#include <stdexcept>  // for invalid_argument
#include <string.h>   // for memset
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(udpPacketGen);

udpPacketGen::udpPacketGen(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&udpPacketGen::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)) {

    if (geometry.header_size < sizeof(udp_packet_headers) + sizeof(udp_app_header))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: header_size {:d} has no room for the {:d} byte application "
                "header after the {:d} byte Ethernet/IP/UDP headers"),
            unique_name, geometry.header_size, sizeof(udp_app_header), sizeof(udp_packet_headers)));
    for (auto [key, mac] : {std::pair{"src_mac", src_mac}, std::pair{"dst_mac", dst_mac}}) {
        const std::string value = config.get<std::string>(unique_name, key);
        if (!parse_mac(value, mac))
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: cannot parse {:s} \"{:s}\""), unique_name, key, value));
    }
    for (auto [key, ip] : {std::pair{"src_ip", &src_ip}, std::pair{"dst_ip", &dst_ip}}) {
        const std::string value = config.get<std::string>(unique_name, key);
        if (!parse_ipv4(value, *ip))
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: cannot parse {:s} \"{:s}\""), unique_name, key, value));
    }
    src_port = config.get<uint16_t>(unique_name, "src_port");
    dst_port = config.get<uint16_t>(unique_name, "dst_port");
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);

    header_buf->register_producer(unique_name);
    payload_buf->register_producer(unique_name);
}

void udpPacketGen::main_thread() {
    frameID frame_id(header_buf);
    std::vector<bool> initialized(header_buf->num_frames, false);
    const uint16_t udp_payload_len =
        geometry.header_size - sizeof(udp_packet_headers) + geometry.payload_size;
    uint64_t seq = 0;
    int produced = 0;

    while (!stop_thread && (num_frames < 0 || produced < num_frames)) {
        uint8_t* headers = header_buf->wait_for_empty_frame(unique_name, frame_id);
        if (headers == nullptr)
            break;
        uint8_t* payload = payload_buf->wait_for_empty_frame(unique_name, frame_id);
        if (payload == nullptr)
            break;

        if (!initialized[frame_id]) {
            for (size_t j = 0; j < geometry.packets_per_frame; j++) {
                uint8_t* slot = headers + j * geometry.header_slot;
                memset(slot, 0, geometry.header_slot);
                fill_udp_headers(reinterpret_cast<udp_packet_headers*>(slot), src_mac, dst_mac,
                                 src_ip, dst_ip, src_port, dst_port, udp_payload_len);
                udp_slot_tail* tail = udp_slot_tail_of(slot, geometry.header_slot);
                tail->wire_len = geometry.wire_len();
                tail->flags = UDP_SLOT_CSUM_OK;
                uint8_t* data = payload + j * geometry.payload_slot;
                for (size_t k = 0; k < geometry.payload_size; k++)
                    data[k] = (j + k) & 0xff;
            }
            initialized[frame_id] = true;
        }
        for (size_t j = 0; j < geometry.packets_per_frame; j++) {
            udp_app_header* app = reinterpret_cast<udp_app_header*>(
                headers + j * geometry.header_slot + sizeof(udp_packet_headers));
            app->seq = seq++;
        }

        header_buf->mark_frame_full(unique_name, frame_id);
        payload_buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
        produced++;
    }
    INFO("Produced {:d} frames of {:d} packets", produced, geometry.packets_per_frame);
}
