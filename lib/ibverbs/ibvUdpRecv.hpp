#ifndef IBV_UDP_RECV_HPP
#define IBV_UDP_RECV_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "ibvUtils.hpp"          // for ibvDevice, ibvFrameMRs
#include "prometheusMetrics.hpp" // for Counter, Gauge
#include "udpPacket.hpp"         // for udp_slot_geometry

#include <infiniband/verbs.h> // for ibv_cq_ex, ibv_qp, ibv_flow, ibv_recv_wr, ibv_sge
#include <stddef.h>           // for size_t
#include <stdint.h>           // for uint8_t, uint32_t, uint64_t
#include <string>             // for string
#include <thread>             // for thread
#include <vector>             // for vector

/**
 * @class ibvUdpRecv
 * @brief Receives UDP packets into a header buffer and a payload buffer with a raw packet QP.
 *
 * A flow steering rule sends every packet for (dst MAC, dst IP, dst port) arriving on the
 * interface to this QP, bypassing the kernel. Each packet is one receive work request with a
 * three-entry scatter list: the first @c header_size bytes land in a header slot, the next
 * @c payload_slot bytes in the matching payload slot, and anything beyond that in a scratch sink,
 * so an oversize packet is flagged instead of putting the QP in error (see udp_slot_geometry).
 * The NIC writes packets in place; the stage only polls the completion queue, records each
 * packet's wire length and checksum result in its header slot tail, and marks the header and
 * payload frames full once all packets_per_frame have arrived. Packets are not reordered or
 * aligned: the consumer reads the headers to find out what it got.
 *
 * queue_depth / packets_per_frame frames are posted ahead, so that much buffer space is always
 * waiting for the NIC. Packets that arrive while no receive work request is posted are dropped by
 * the NIC and show up in its out_of_buffer counter.
 *
 * Raw packet QPs need CAP_NET_RAW: run `setcap cap_net_raw+ep <build>/kotekan/kotekan` after
 * every build, or run as root.
 *
 * @par Buffers
 * @buffer header_buf   Header slots, produced in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes, each ending in a
 *                        udp_slot_tail
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf interface       String. Netdev to receive on, e.g. enp1s0f0np0.
 * @conf dst_ip          String. Destination IPv4 address to steer, normally the interface's.
 * @conf dst_port        Int. Destination UDP port to steer.
 * @conf dst_mac         String, default the interface MAC. Destination MAC to steer; set it for
 *                       multicast.
 * @conf header_size     Int. See udp_slot_geometry.
 * @conf header_slot     Int. See udp_slot_geometry.
 * @conf payload_size    Int. See udp_slot_geometry.
 * @conf payload_slot    Int. See udp_slot_geometry.
 * @conf queue_depth     Int, default 8192. Receive work requests the QP holds: at least
 *                       packets_per_frame and at most the device's max_qp_wr.
 * @conf poll_mode       String, default "event". "event" sleeps on the completion channel between
 *                       batches, with the CQ moderated to one event per cq_moderation_count
 *                       completions or cq_moderation_period_us; "busy" spins on the CQ.
 * @conf cq_moderation_count      Int, default 256.
 * @conf cq_moderation_period_us  Int, default 200.
 *
 * @par Metrics
 * @metric kotekan_ibv_udp_recv_packets_total            Packets received.
 * @metric kotekan_ibv_udp_recv_bytes_total              Bytes received, FCS excluded.
 * @metric kotekan_ibv_udp_recv_frames_total             Frames filled.
 * @metric kotekan_ibv_udp_recv_bad_length_packets_total Packets whose wire length is not
 *                                                       header_size + payload_size.
 * @metric kotekan_ibv_udp_recv_csum_errors_total        Packets the NIC flagged with a bad IP or
 *                                                       UDP checksum.
 * @metric kotekan_ibv_udp_recv_out_of_buffer            Packets the NIC dropped for lack of a
 *                                                       posted work request since the stage
 *                                                       started. Device-wide: every raw QP on the
 *                                                       device counts here.
 *
 * @author Andre Renard
 */
class ibvUdpRecv : public kotekan::Stage {
public:
    ibvUdpRecv(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~ibvUdpRecv() override;
    void main_thread() override;

private:
    /// Waits for empty header and payload frames @p frame_id and posts a receive work request
    /// per packet. Returns false on shutdown.
    bool post_frame(int frame_id);
    /// Polls up to packets_per_frame completions, filling slot tails and marking full frames.
    /// Returns how many completions it took.
    int poll_completions();
    /// Updates the out_of_buffer gauge once a second until the stage stops. Runs in its own
    /// thread: the sysfs read is a firmware query, too slow for the receive loop.
    void track_out_of_buffer();
    /// Destroys the flow, QP, CQ, completion channel and sink; the destructor and a failed
    /// constructor.
    void release();

    Buffer* header_buf;
    Buffer* payload_buf;
    const udp_slot_geometry geometry;
    ibvDevice device;
    ibvFrameMRs header_mrs;
    ibvFrameMRs payload_mrs;

    /// Scratch that the tail of an oversize packet scatters into.
    uint8_t* sink = nullptr;
    size_t sink_len;
    ibv_mr* sink_mr = nullptr;

    ibv_comp_channel* channel = nullptr;
    ibv_cq_ex* cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_flow* flow = nullptr;
    struct {
        ibv_flow_attr attr;
        ibv_flow_spec_eth eth;
        ibv_flow_spec_ipv4 ip;
        ibv_flow_spec_tcp_udp udp;
    } __attribute__((packed)) flow_rule;

    uint32_t queue_depth;
    int frames_in_flight;
    bool event_mode;
    bool cq_armed = false;

    /// One receive work request per packet of a frame, chained, reused for every post.
    std::vector<ibv_recv_wr> wrs;
    std::vector<ibv_sge> sges;
    uint64_t next_wr_id = 0;
    uint64_t expected_wr_id = 0;

    /// The frame being filled, how many of its packets have arrived, and how many frames
    /// completed since the last post.
    int fill_frame_id = 0;
    size_t fill_count = 0;
    int frames_to_post = 0;

    std::string out_of_buffer_path;
    uint64_t out_of_buffer_base = 0;
    std::thread stats_thread;

    kotekan::prometheus::Counter& packets_counter;
    kotekan::prometheus::Counter& bytes_counter;
    kotekan::prometheus::Counter& frames_counter;
    kotekan::prometheus::Counter& bad_length_counter;
    kotekan::prometheus::Counter& csum_error_counter;
    kotekan::prometheus::Gauge& out_of_buffer_gauge;
};

#endif /* IBV_UDP_RECV_HPP */
