#include "ibvUdpSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "ibvUtils.hpp"          // for raw_qp_to_state, create_nonblocking_comp_channel, wait_f...
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "udpPacket.hpp"         // for udp_packet_headers, ETHER_TYPE_IPV4
#include "visUtil.hpp"           // for frameID, current_time, double_to_ts

#include "fmt.hpp" // for format

#include <algorithm>   // for min
#include <arpa/inet.h> // for htons, ntohs
#include <errno.h>     // for errno, EPERM, ENOENT
#include <functional>  // for bind
#include <stdexcept>   // for runtime_error, invalid_argument
#include <string.h>    // for strerror, memset

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(ibvUdpSend);

ibvUdpSend::ibvUdpSend(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ibvUdpSend::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)),
    device(config.get<std::string>(unique_name, "interface")),
    header_mrs(device.pd, header_buf, IBV_ACCESS_RELAXED_ORDERING),
    payload_mrs(device.pd, payload_buf, IBV_ACCESS_RELAXED_ORDERING),
    packets_counter(
        Metrics::instance().add_counter("kotekan_ibv_udp_send_packets_total", unique_name)),
    bytes_counter(Metrics::instance().add_counter("kotekan_ibv_udp_send_bytes_total", unique_name)),
    frames_counter(
        Metrics::instance().add_counter("kotekan_ibv_udp_send_frames_total", unique_name)) {
    try {
        queue_depth = config.get_default<uint32_t>(unique_name, "queue_depth", 8192);
        const uint32_t max_qp_wr = device.attr.orig_attr.max_qp_wr;
        if (queue_depth > max_qp_wr || queue_depth < geometry.packets_per_frame)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: queue_depth {:d} must be between packets_per_frame {:d} and "
                                "the device limit {:d}"),
                            unique_name, queue_depth, geometry.packets_per_frame, max_qp_wr));
        max_frames_in_flight =
            std::min((int)(queue_depth / geometry.packets_per_frame), header_buf->num_frames);
        if (geometry.wire_len() - sizeof(ethernet_header) > device.mtu)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: packets of {:d} bytes exceed the MTU of {:s} ({:d})"),
                            unique_name, geometry.wire_len(), device.netdev, device.mtu));
        if (!(device.attr.raw_packet_caps & IBV_RAW_PACKET_CAP_IP_CSUM))
            throw std::runtime_error(
                fmt::format(fmt("{:s}: {:s} cannot offload IP checksums on raw packet QPs"),
                            unique_name, device.rdma_name));

        channel = create_nonblocking_comp_channel(device.ctx);

        ibv_cq_init_attr_ex cq_attr;
        memset(&cq_attr, 0, sizeof(cq_attr));
        cq_attr.cqe = queue_depth;
        cq_attr.channel = channel;
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED;
        cq = ibv_create_cq_ex(device.ctx, &cq_attr);
        if (!cq)
            throw std::runtime_error(fmt::format(fmt("{:s}: ibv_create_cq_ex failed: {:s}"),
                                                 unique_name, strerror(errno)));

        ibv_qp_init_attr_ex qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq = ibv_cq_ex_to_cq(cq);
        qp_attr.recv_cq = ibv_cq_ex_to_cq(cq);
        qp_attr.cap.max_send_wr = queue_depth;
        qp_attr.cap.max_send_sge = 2;
        qp_attr.qp_type = IBV_QPT_RAW_PACKET;
        qp_attr.sq_sig_all = 0;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        qp_attr.pd = device.pd;
        qp_attr.send_ops_flags = IBV_QP_EX_WITH_SEND;
        qp = ibv_create_qp_ex(device.ctx, &qp_attr);
        if (!qp)
            throw std::runtime_error(fmt::format(
                fmt("{:s}: ibv_create_qp_ex failed: {:s}{:s}"), unique_name, strerror(errno),
                errno == EPERM ? " (raw packet QPs need CAP_NET_RAW: setcap cap_net_raw+ep on the "
                                 "kotekan binary, or run as root)"
                               : ""));
        qpx = ibv_qp_to_qp_ex(qp);
        raw_qp_to_state(qp, device.port, IBV_QPS_RTS);

        const uint32_t rate_limit_kbps =
            config.get_default<uint32_t>(unique_name, "rate_limit_kbps", 0);
        if (rate_limit_kbps) {
            ibv_qp_rate_limit_attr rate;
            memset(&rate, 0, sizeof(rate));
            rate.rate_limit = rate_limit_kbps;
            rate.typical_pkt_sz = geometry.wire_len();
            const int rc = ibv_modify_qp_rate_limit(qp, &rate);
            if (rc)
                throw std::runtime_error(
                    fmt::format(fmt("{:s}: ibv_modify_qp_rate_limit({:d} kbps) failed: {:s}"),
                                unique_name, rate_limit_kbps, strerror(rc)));
        }

        header_buf->register_consumer(unique_name);
        payload_buf->register_consumer(unique_name);
    } catch (...) {
        release();
        throw;
    }
}

ibvUdpSend::~ibvUdpSend() {
    release();
}

void ibvUdpSend::release() {
    if (qp && ibv_destroy_qp(qp))
        ERROR("ibv_destroy_qp failed: {:s}", strerror(errno));
    if (cq && ibv_destroy_cq(ibv_cq_ex_to_cq(cq)))
        ERROR("ibv_destroy_cq failed: {:s}", strerror(errno));
    if (channel && ibv_destroy_comp_channel(channel))
        ERROR("ibv_destroy_comp_channel failed: {:s}", strerror(errno));
    qp = nullptr;
    cq = nullptr;
    channel = nullptr;
}

void ibvUdpSend::main_thread() {
    INFO("Sending {:d} byte packets on {:s} ({:s}), {:d} per frame, up to {:d} frames in flight",
         geometry.wire_len(), device.netdev, device.rdma_name, geometry.packets_per_frame,
         max_frames_in_flight);

    frameID frame_id(header_buf);
    while (!stop_thread) {
        const int retired = retire_frames();
        if ((int)in_flight.size() >= max_frames_in_flight) {
            // The QP is full: sleep on the completion channel until a frame retires. Arm the
            // CQ, then poll once more before blocking (see wait_for_cq_event).
            if (retired)
                continue;
            if (!cq_armed) {
                if (ibv_req_notify_cq(ibv_cq_ex_to_cq(cq), 0))
                    FATAL_ERROR("ibv_req_notify_cq failed: {:s}", strerror(errno));
                cq_armed = true;
                continue;
            }
            if (wait_for_cq_event(channel, 100))
                cq_armed = false;
            continue;
        }
        // Timed wait so completed frames keep retiring while the producer is slow.
        const int status = header_buf->wait_for_full_frame_timeout(
            unique_name, frame_id, double_to_ts(current_time() + 0.1));
        if (status == -1)
            break;
        if (status == 1)
            continue;
        if (payload_buf->wait_for_full_frame(unique_name, frame_id) == nullptr)
            break;
        post_frame(frame_id);
        frame_id++;
    }
}

int ibvUdpSend::retire_frames() {
    ibv_poll_cq_attr poll_attr;
    memset(&poll_attr, 0, sizeof(poll_attr));
    int rc = ibv_start_poll(cq, &poll_attr);
    if (rc == ENOENT)
        return 0;
    if (rc)
        FATAL_ERROR("ibv_start_poll failed: {:s}", strerror(rc));
    int retired = 0;
    do {
        if (cq->status != IBV_WC_SUCCESS) {
            if (stop_thread && cq->status == IBV_WC_WR_FLUSH_ERR)
                break;
            FATAL_ERROR("Send on {:s} failed: {:s} (wr_id {:d})", device.netdev,
                        ibv_wc_status_str(cq->status), cq->wr_id);
        }
        if (in_flight.empty() || cq->wr_id != in_flight.front().second)
            FATAL_ERROR("Unexpected send completion wr_id {:d}, expected {:d}", cq->wr_id,
                        in_flight.empty() ? -1 : (int64_t)in_flight.front().second);
        header_buf->mark_frame_empty(unique_name, in_flight.front().first);
        payload_buf->mark_frame_empty(unique_name, in_flight.front().first);
        in_flight.pop_front();
        retired++;
    } while ((rc = ibv_next_poll(cq)) == 0);
    ibv_end_poll(cq);
    if (rc != ENOENT && rc != 0)
        FATAL_ERROR("ibv_next_poll failed: {:s}", strerror(rc));
    frames_counter.inc(retired);
    return retired;
}

void ibvUdpSend::post_frame(int frame_id) {
    uint8_t* headers = header_buf->frames[frame_id];
    uint8_t* payload = payload_buf->frames[frame_id];
    const uint32_t header_lkey = header_mrs.lkey(frame_id);
    const uint32_t payload_lkey = payload_mrs.lkey(frame_id);

    // An IP length that disagrees with the slot geometry means the producer and this stage
    // were configured with different packet sizes.
    const udp_packet_headers* first = reinterpret_cast<const udp_packet_headers*>(headers);
    if (first->eth.ether_type == htons(ETHER_TYPE_IPV4)
        && ntohs(first->ip.total_length) != geometry.wire_len() - sizeof(ethernet_header))
        FATAL_ERROR("Frame {:d}: the IP total length {:d} does not match header_size {:d} + "
                    "payload_size {:d}",
                    frame_id, ntohs(first->ip.total_length), geometry.header_size,
                    geometry.payload_size);

    ibv_sge sge[2];
    sge[0].length = geometry.header_size;
    sge[0].lkey = header_lkey;
    sge[1].length = geometry.payload_size;
    sge[1].lkey = payload_lkey;

    ibv_wr_start(qpx);
    for (size_t i = 0; i < geometry.packets_per_frame; i++) {
        qpx->wr_id = next_wr_id++;
        qpx->wr_flags =
            IBV_SEND_IP_CSUM | (i + 1 == geometry.packets_per_frame ? IBV_SEND_SIGNALED : 0);
        ibv_wr_send(qpx);
        sge[0].addr = (uintptr_t)(headers + i * geometry.header_slot);
        sge[1].addr = (uintptr_t)(payload + i * geometry.payload_slot);
        ibv_wr_set_sge_list(qpx, 2, sge);
    }
    const int rc = ibv_wr_complete(qpx);
    if (rc)
        FATAL_ERROR("ibv_wr_complete failed for frame {:d}: {:s}", frame_id, strerror(rc));

    in_flight.emplace_back(frame_id, next_wr_id - 1);
    packets_counter.inc(geometry.packets_per_frame);
    bytes_counter.inc(geometry.packets_per_frame * geometry.wire_len());
}
