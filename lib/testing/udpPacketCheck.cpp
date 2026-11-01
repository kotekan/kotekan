#include "udpPacketCheck.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "errors.h"              // for exit_kotekan, CLEAN_EXIT, TEST_FAILED
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "udpPacket.hpp"         // for udp_packet_headers, udp_app_header, udp_slot_tail_of
#include "visUtil.hpp"           // for frameID

#include "fmt.hpp" // for format

#include <arpa/inet.h> // for htons
#include <functional>  // for bind
#include <stdexcept>   // for invalid_argument

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(udpPacketCheck);

udpPacketCheck::udpPacketCheck(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&udpPacketCheck::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)),
    packets_counter(
        Metrics::instance().add_counter("kotekan_udp_packet_check_packets_total", unique_name)),
    bad_packets_counter(
        Metrics::instance().add_counter("kotekan_udp_packet_check_bad_packets_total", unique_name)),
    seq_gaps_counter(
        Metrics::instance().add_counter("kotekan_udp_packet_check_seq_gaps_total", unique_name)),
    lost_packets_counter(Metrics::instance().add_counter(
        "kotekan_udp_packet_check_lost_packets_total", unique_name)),
    reordered_packets_counter(Metrics::instance().add_counter(
        "kotekan_udp_packet_check_reordered_packets_total", unique_name)) {

    if (geometry.header_size < sizeof(udp_packet_headers) + sizeof(udp_app_header))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: header_size {:d} has no room for the {:d} byte application "
                "header after the {:d} byte Ethernet/IP/UDP headers"),
            unique_name, geometry.header_size, sizeof(udp_app_header), sizeof(udp_packet_headers)));
    dst_port = config.get<uint16_t>(unique_name, "dst_port");
    payload_check_interval = config.get_default<size_t>(unique_name, "payload_check_interval", 64);
    fatal_on_bad_packet = config.get_default<bool>(unique_name, "fatal_on_bad_packet", true);
    report_interval = config.get_default<int>(unique_name, "report_interval", 100);
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    max_lost_packets = config.get_default<int64_t>(unique_name, "max_lost_packets", -1);

    header_buf->register_consumer(unique_name);
    payload_buf->register_consumer(unique_name);
}

void udpPacketCheck::main_thread() {
    frameID frame_id(header_buf);
    uint64_t packet_index = 0;
    int checked = 0;

    while (!stop_thread && (num_frames < 0 || checked < num_frames)) {
        uint8_t* headers = header_buf->wait_for_full_frame(unique_name, frame_id);
        if (headers == nullptr)
            break;
        uint8_t* payload = payload_buf->wait_for_full_frame(unique_name, frame_id);
        if (payload == nullptr)
            break;

        for (size_t j = 0; j < geometry.packets_per_frame; j++, packet_index++) {
            const std::string problem =
                check_packet(headers + j * geometry.header_slot,
                             payload + j * geometry.payload_slot, j, packet_index);
            if (problem.empty())
                continue;
            bad_packets++;
            if (fatal_on_bad_packet)
                FATAL_ERROR("Frame {:d} packet {:d}: {:s}", frame_id, j, problem);
            ERROR("Frame {:d} packet {:d}: {:s}", frame_id, j, problem);
        }
        packets_counter.inc(geometry.packets_per_frame);

        header_buf->mark_frame_empty(unique_name, frame_id);
        payload_buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
        checked++;
        if (report_interval > 0 && checked % report_interval == 0)
            INFO("{:d} frames, {:d} packets: {:d} bad, {:d} gaps losing {:d} packets, {:d} "
                 "reordered",
                 checked, packet_index, bad_packets, seq_gaps, lost_packets, reordered_packets);
    }

    if (num_frames >= 0 && checked >= num_frames) {
        INFO("Checked {:d} frames, {:d} packets: {:d} bad, {:d} gaps losing {:d} packets, {:d} "
             "reordered",
             checked, packet_index, bad_packets, seq_gaps, lost_packets, reordered_packets);
        if (bad_packets > 0 || (max_lost_packets >= 0 && (int64_t)lost_packets > max_lost_packets))
            TEST_FAILED();
        else
            exit_kotekan(CLEAN_EXIT);
    }
}

std::string udpPacketCheck::check_packet(const uint8_t* header, const uint8_t* payload, size_t slot,
                                         uint64_t packet_index) {
    const udp_slot_tail* tail =
        udp_slot_tail_of(const_cast<uint8_t*>(header), geometry.header_slot);
    if (tail->wire_len != geometry.wire_len())
        return fmt::format(fmt("wire length {:d}, expected {:d}"), tail->wire_len,
                           geometry.wire_len());
    if (!(tail->flags & UDP_SLOT_CSUM_OK))
        return "bad IP or UDP checksum";
    const udp_packet_headers* h = reinterpret_cast<const udp_packet_headers*>(header);
    if (h->eth.ether_type != htons(ETHER_TYPE_IPV4))
        return fmt::format(fmt("EtherType {:#06x} is not IPv4"), ntohs(h->eth.ether_type));
    if (h->ip.protocol != IP_PROTOCOL_UDP)
        return fmt::format(fmt("IP protocol {:d} is not UDP"), h->ip.protocol);
    if (h->udp.dest_port != htons(dst_port))
        return fmt::format(fmt("UDP destination port {:d}, expected {:d}"), ntohs(h->udp.dest_port),
                           dst_port);

    const uint64_t seq =
        reinterpret_cast<const udp_app_header*>(header + sizeof(udp_packet_headers))->seq;
    if (!have_seq) {
        have_seq = true;
        expected_seq = seq + 1;
    } else if (seq == expected_seq) {
        expected_seq++;
    } else if (seq > expected_seq) {
        seq_gaps++;
        seq_gaps_counter.inc();
        lost_packets += seq - expected_seq;
        lost_packets_counter.inc(seq - expected_seq);
        expected_seq = seq + 1;
    } else {
        reordered_packets++;
        reordered_packets_counter.inc();
    }

    if (payload_check_interval == 0 || packet_index % payload_check_interval != 0)
        return "";
    // udpPacketGen filled this payload for the slot its sequence number maps to.
    const size_t gen_slot = seq % geometry.packets_per_frame;
    for (size_t k = 0; k < geometry.payload_size; k++) {
        if (payload[k] != ((gen_slot + k) & 0xff))
            return fmt::format(fmt("payload byte {:d} is {:#04x}, expected {:#04x} (seq {:d}, "
                                   "received in slot {:d})"),
                               k, payload[k], (gen_slot + k) & 0xff, seq, slot);
    }
    return "";
}
