#ifndef UDP_PACKET_GEN_HPP
#define UDP_PACKET_GEN_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "udpPacket.hpp"       // for udp_slot_geometry

#include <stdint.h> // for uint8_t, uint16_t, uint32_t
#include <string>   // for string

/**
 * @class udpPacketGen
 * @brief Fills a header buffer and a payload buffer with test packets for ibvUdpSend.
 *
 * Every header slot gets Ethernet/IPv4/UDP headers built from the config, a udp_app_header with
 * a running sequence number, and the udp_slot_tail ibvUdpRecv would write. Payload slot j is
 * filled with the byte pattern ((j + k) & 0xff) at offset k the first time its frame is used.
 * Later passes over the ring store only the sequence number, so the stage costs one 8-byte store
 * per packet and cannot be the bottleneck at line rate. udpPacketCheck verifies the result.
 *
 * The sequence number of the packet in slot j is always congruent to j modulo packets_per_frame,
 * which is how udpPacketCheck knows which pattern a received payload should carry. Buffers that
 * zero their frames on release (@c zero_frames) would wipe the headers between passes; leave it
 * off.
 *
 * @par Buffers
 * @buffer header_buf   Header slots, produced in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf src_mac       String. Source MAC, e.g. "30:c5:99:3f:09:1e".
 * @conf dst_mac       String. Destination MAC.
 * @conf src_ip        String. Source IPv4 address.
 * @conf dst_ip        String. Destination IPv4 address.
 * @conf src_port      Int. Source UDP port.
 * @conf dst_port      Int. Destination UDP port.
 * @conf header_size   Int. See udp_slot_geometry; at least 50 here (the headers plus the
 *                     8 byte udp_app_header).
 * @conf header_slot   Int. See udp_slot_geometry.
 * @conf payload_size  Int. See udp_slot_geometry.
 * @conf payload_slot  Int. See udp_slot_geometry.
 * @conf num_frames    Int, default -1 (unlimited). Frames to produce before the stage exits.
 *
 * @author Andre Renard
 */
class udpPacketGen : public kotekan::Stage {
public:
    udpPacketGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~udpPacketGen() override = default;
    void main_thread() override;

private:
    Buffer* header_buf;
    Buffer* payload_buf;
    const udp_slot_geometry geometry;
    uint8_t src_mac[6];
    uint8_t dst_mac[6];
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    int num_frames;
};

#endif /* UDP_PACKET_GEN_HPP */
