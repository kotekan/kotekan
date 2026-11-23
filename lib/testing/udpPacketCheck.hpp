#ifndef UDP_PACKET_CHECK_HPP
#define UDP_PACKET_CHECK_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter
#include "udpPacket.hpp"         // for udp_slot_geometry

#include <stdint.h> // for uint8_t, uint16_t, uint64_t
#include <string>   // for string

/**
 * @class udpPacketCheck
 * @brief Checks the packets in a header buffer and a payload buffer that udpPacketGen produced,
 *        directly or through ibvUdpSend and ibvUdpRecv.
 *
 * For every packet: the slot tail holds header_size + payload_size bytes with the checksum flag
 * set; the Ethernet type, IP protocol and UDP destination port are as expected; the sequence
 * number in the udp_app_header is tracked for gaps, repeats and reordering; and every
 * payload_check_interval'th packet carries the udpPacketGen pattern. A bad packet is a
 * FATAL_ERROR unless fatal_on_bad_packet is false. Sequence gaps are counted, not fatal: lost
 * packets are what a network test measures.
 *
 * @par Buffers
 * @buffer header_buf   Header slots, consumed in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf dst_port               Int. The UDP destination port every packet must carry.
 * @conf header_size            Int. See udp_slot_geometry; at least 50.
 * @conf header_slot            Int. See udp_slot_geometry.
 * @conf payload_size           Int. See udp_slot_geometry.
 * @conf payload_slot           Int. See udp_slot_geometry.
 * @conf payload_check_interval Int, default 64. Verify the payload of one packet in this many;
 *                              0 verifies none.
 * @conf fatal_on_bad_packet    Bool, default true.
 * @conf report_interval        Int, default 100. Log the counters every this many frames.
 * @conf num_frames             Int, default -1 (forever). Frames to check before exiting kotekan.
 * @conf max_lost_packets       Int, default -1 (unlimited). With num_frames, exit with
 *                              TEST_FAILED when more packets than this were lost.
 *
 * @par Metrics
 * @metric kotekan_udp_packet_check_packets_total            Packets checked.
 * @metric kotekan_udp_packet_check_bad_packets_total        Packets with a bad length, header,
 *                                                           checksum flag or payload.
 * @metric kotekan_udp_packet_check_seq_gaps_total           Jumps forward in the sequence.
 * @metric kotekan_udp_packet_check_lost_packets_total       Sequence numbers skipped by the gaps.
 * @metric kotekan_udp_packet_check_reordered_packets_total  Packets whose sequence number was
 *                                                           already passed.
 *
 * @author Andre Renard
 */
class udpPacketCheck : public kotekan::Stage {
public:
    udpPacketCheck(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~udpPacketCheck() override = default;
    void main_thread() override;

private:
    /// Checks one packet; returns a description of the first problem, or an empty string.
    std::string check_packet(const uint8_t* header, const uint8_t* payload, size_t slot,
                             uint64_t packet_index);

    Buffer* header_buf;
    Buffer* payload_buf;
    const udp_slot_geometry geometry;
    uint16_t dst_port;
    size_t payload_check_interval;
    bool fatal_on_bad_packet;
    int report_interval;
    int num_frames;
    int64_t max_lost_packets;

    bool have_seq = false;
    uint64_t expected_seq = 0;
    uint64_t bad_packets = 0;
    uint64_t seq_gaps = 0;
    uint64_t lost_packets = 0;
    uint64_t reordered_packets = 0;

    kotekan::prometheus::Counter& packets_counter;
    kotekan::prometheus::Counter& bad_packets_counter;
    kotekan::prometheus::Counter& seq_gaps_counter;
    kotekan::prometheus::Counter& lost_packets_counter;
    kotekan::prometheus::Counter& reordered_packets_counter;
};

#endif /* UDP_PACKET_CHECK_HPP */
