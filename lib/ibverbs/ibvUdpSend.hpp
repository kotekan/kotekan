#ifndef IBV_UDP_SEND_HPP
#define IBV_UDP_SEND_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "ibvUtils.hpp"          // for ibvDevice, ibvFrameMRs
#include "prometheusMetrics.hpp" // for Counter
#include "udpPacket.hpp"         // for udp_slot_geometry

#include <deque>              // for deque
#include <infiniband/verbs.h> // for ibv_cq_ex, ibv_qp, ibv_qp_ex, ibv_comp_channel
#include <stdint.h>           // for uint32_t, uint64_t
#include <string>             // for string
#include <utility>            // for pair

/**
 * @class ibvUdpSend
 * @brief Sends the packets held in a header buffer and a payload buffer with a raw packet QP.
 *
 * Packet j of frame i is the first @c header_size bytes of header slot j followed by the first
 * @c payload_size bytes of payload slot j (see udp_slot_geometry). Every packet is one send work
 * request with a two-entry gather list, so the NIC reads headers and payload in place and the
 * stage never touches packet bytes. The header slots must hold complete, untagged
 * Ethernet/IPv4/UDP headers (udpPacketGen or fill_udp_headers write them); the NIC fills in the
 * IP and UDP checksums.
 *
 * A frame's work requests are posted with one doorbell and only the last one asks for a
 * completion; that completion retires the frame, and both frames are marked empty. Up to
 * queue_depth / packets_per_frame frames are in flight at once.
 *
 * Raw packet QPs need CAP_NET_RAW: run `setcap cap_net_raw+ep <build>/kotekan/kotekan` after
 * every build, or run as root.
 *
 * @par Buffers
 * @buffer header_buf   Header slots, consumed in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf interface       String. Netdev to send on, e.g. enp1s0f0np0.
 * @conf header_size     Int. See udp_slot_geometry.
 * @conf header_slot     Int. See udp_slot_geometry.
 * @conf payload_size    Int. See udp_slot_geometry.
 * @conf payload_slot    Int. See udp_slot_geometry.
 * @conf queue_depth     Int, default 8192. Send work requests the QP holds: at least
 *                       packets_per_frame and at most the device's max_qp_wr.
 * @conf rate_limit_kbps Int, default 0 (off). Packet pacing rate of the QP in kbit/s.
 *
 * @par Metrics
 * @metric kotekan_ibv_udp_send_packets_total  Packets posted.
 * @metric kotekan_ibv_udp_send_bytes_total    Bytes posted, FCS excluded.
 * @metric kotekan_ibv_udp_send_frames_total   Frames sent and released.
 *
 * @author Andre Renard
 */
class ibvUdpSend : public kotekan::Stage {
public:
    ibvUdpSend(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~ibvUdpSend() override;
    void main_thread() override;

private:
    /// Polls the CQ and releases the frames whose last packet completed. Returns how many.
    int retire_frames();
    /// Posts every packet of frame @p frame_id with one doorbell.
    void post_frame(int frame_id);
    /// Destroys the QP, CQ and completion channel; the destructor and a failed constructor.
    void release();

    Buffer* header_buf;
    Buffer* payload_buf;
    const udp_slot_geometry geometry;
    ibvDevice device;
    ibvFrameMRs header_mrs;
    ibvFrameMRs payload_mrs;

    ibv_comp_channel* channel = nullptr;
    ibv_cq_ex* cq = nullptr;
    ibv_qp* qp = nullptr;
    ibv_qp_ex* qpx = nullptr;

    uint32_t queue_depth;
    int max_frames_in_flight;
    uint64_t next_wr_id = 0;
    /// Frames posted and not yet retired, oldest first, with the wr_id of their last packet.
    std::deque<std::pair<int, uint64_t>> in_flight;
    bool cq_armed = false;

    kotekan::prometheus::Counter& packets_counter;
    kotekan::prometheus::Counter& bytes_counter;
    kotekan::prometheus::Counter& frames_counter;
};

#endif /* IBV_UDP_SEND_HPP */
