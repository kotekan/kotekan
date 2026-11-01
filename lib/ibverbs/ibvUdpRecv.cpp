#include "ibvUdpRecv.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "ibvUtils.hpp"          // for raw_qp_to_state, create_nonblocking_comp_channel, wait_f...
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR, WARN, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "udpPacket.hpp"         // for udp_slot_tail_of, parse_mac, parse_ipv4, UDP_SLOT_CSUM_OK
#include "visUtil.hpp"           // for frameID

#include "fmt.hpp" // for format

#include <algorithm>   // for min
#include <arpa/inet.h> // for htons
#include <errno.h>     // for errno, EPERM, ENOENT
#include <fstream>     // for ifstream
#include <functional>  // for bind
#include <stdexcept>   // for runtime_error, invalid_argument
#include <stdlib.h>    // for posix_memalign, free
#include <string.h>    // for strerror, memset
#include <unistd.h>    // for usleep

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(ibvUdpRecv);

ibvUdpRecv::ibvUdpRecv(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ibvUdpRecv::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)),
    device(config.get<std::string>(unique_name, "interface")),
    header_mrs(device.pd, header_buf, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING),
    payload_mrs(device.pd, payload_buf, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING),
    packets_counter(
        Metrics::instance().add_counter("kotekan_ibv_udp_recv_packets_total", unique_name)),
    bytes_counter(Metrics::instance().add_counter("kotekan_ibv_udp_recv_bytes_total", unique_name)),
    frames_counter(
        Metrics::instance().add_counter("kotekan_ibv_udp_recv_frames_total", unique_name)),
    bad_length_counter(Metrics::instance().add_counter(
        "kotekan_ibv_udp_recv_bad_length_packets_total", unique_name)),
    csum_error_counter(
        Metrics::instance().add_counter("kotekan_ibv_udp_recv_csum_errors_total", unique_name)),
    out_of_buffer_gauge(
        Metrics::instance().add_gauge("kotekan_ibv_udp_recv_out_of_buffer", unique_name)) {
    try {
        queue_depth = config.get_default<uint32_t>(unique_name, "queue_depth", 8192);
        const uint32_t max_qp_wr = device.attr.orig_attr.max_qp_wr;
        if (queue_depth > max_qp_wr || queue_depth < geometry.packets_per_frame)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: queue_depth {:d} must be between packets_per_frame {:d} and "
                                "the device limit {:d}"),
                            unique_name, queue_depth, geometry.packets_per_frame, max_qp_wr));
        frames_in_flight =
            std::min((int)(queue_depth / geometry.packets_per_frame), header_buf->num_frames);

        const std::string poll_mode =
            config.get_default<std::string>(unique_name, "poll_mode", "event");
        if (poll_mode != "event" && poll_mode != "busy")
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: poll_mode must be \"event\" or \"busy\", not \"{:s}\""),
                            unique_name, poll_mode));
        event_mode = poll_mode == "event";

        // The flow steering rule: untagged IPv4/UDP to (dst MAC, dst IP, dst port).
        uint8_t dst_mac[6];
        memcpy(dst_mac, device.mac, 6);
        const std::string dst_mac_str = config.get_default<std::string>(unique_name, "dst_mac", "");
        if (!dst_mac_str.empty() && !parse_mac(dst_mac_str, dst_mac))
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: cannot parse dst_mac \"{:s}\""), unique_name, dst_mac_str));
        uint32_t dst_ip;
        const std::string dst_ip_str = config.get<std::string>(unique_name, "dst_ip");
        if (!parse_ipv4(dst_ip_str, dst_ip))
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: cannot parse dst_ip \"{:s}\""), unique_name, dst_ip_str));
        const uint16_t dst_port = config.get<uint16_t>(unique_name, "dst_port");
        memset(&flow_rule, 0, sizeof(flow_rule));
        flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
        flow_rule.attr.size = sizeof(flow_rule);
        flow_rule.attr.num_of_specs = 3;
        flow_rule.attr.port = device.port;
        flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
        flow_rule.eth.size = sizeof(flow_rule.eth);
        memcpy(flow_rule.eth.val.dst_mac, dst_mac, 6);
        memset(flow_rule.eth.mask.dst_mac, 0xff, 6);
        flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
        flow_rule.ip.size = sizeof(flow_rule.ip);
        flow_rule.ip.val.dst_ip = dst_ip;
        flow_rule.ip.mask.dst_ip = 0xffffffff;
        flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
        flow_rule.udp.size = sizeof(flow_rule.udp);
        flow_rule.udp.val.dst_port = htons(dst_port);
        flow_rule.udp.mask.dst_port = 0xffff;

        // Anything past header_size + payload_slot scatters here, so an oversize packet completes
        // with its true length instead of a local length error that would stop the QP.
        sink_len = (device.mtu + 18 + 4095) / 4096 * 4096;
        if (posix_memalign((void**)&sink, 4096, sink_len))
            throw std::runtime_error(
                fmt::format(fmt("{:s}: cannot allocate the sink"), unique_name));
        sink_mr = ibv_reg_mr(device.pd, sink, sink_len, IBV_ACCESS_LOCAL_WRITE);
        if (!sink_mr)
            throw std::runtime_error(fmt::format(fmt("{:s}: ibv_reg_mr failed for the sink: {:s}"),
                                                 unique_name, strerror(errno)));

        channel = create_nonblocking_comp_channel(device.ctx);

        ibv_cq_init_attr_ex cq_attr;
        memset(&cq_attr, 0, sizeof(cq_attr));
        cq_attr.cqe = queue_depth;
        cq_attr.channel = channel;
        cq_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN;
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED;
        cq = ibv_create_cq_ex(device.ctx, &cq_attr);
        if (!cq)
            throw std::runtime_error(fmt::format(fmt("{:s}: ibv_create_cq_ex failed: {:s}"),
                                                 unique_name, strerror(errno)));
        if (event_mode) {
            ibv_modify_cq_attr moderate;
            memset(&moderate, 0, sizeof(moderate));
            moderate.attr_mask = IBV_CQ_ATTR_MODERATE;
            moderate.moderate.cq_count =
                config.get_default<uint16_t>(unique_name, "cq_moderation_count", 256);
            moderate.moderate.cq_period =
                config.get_default<uint16_t>(unique_name, "cq_moderation_period_us", 200);
            const int rc = ibv_modify_cq(ibv_cq_ex_to_cq(cq), &moderate);
            if (rc)
                WARN("CQ moderation is not available ({:s}); event mode will wake up per packet",
                     strerror(rc));
        }

        ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq = ibv_cq_ex_to_cq(cq);
        qp_attr.recv_cq = ibv_cq_ex_to_cq(cq);
        qp_attr.cap.max_recv_wr = queue_depth;
        qp_attr.cap.max_recv_sge = 3;
        qp_attr.qp_type = IBV_QPT_RAW_PACKET;
        qp = ibv_create_qp(device.pd, &qp_attr);
        if (!qp)
            throw std::runtime_error(fmt::format(
                fmt("{:s}: ibv_create_qp failed: {:s}{:s}"), unique_name, strerror(errno),
                errno == EPERM ? " (raw packet QPs need CAP_NET_RAW: setcap cap_net_raw+ep on the "
                                 "kotekan binary, or run as root)"
                               : ""));
        raw_qp_to_state(qp, device.port, IBV_QPS_RTR);

        wrs.resize(geometry.packets_per_frame);
        sges.resize(3 * geometry.packets_per_frame);
        for (size_t i = 0; i < geometry.packets_per_frame; i++) {
            memset(&wrs[i], 0, sizeof(wrs[i]));
            wrs[i].next = i + 1 < geometry.packets_per_frame ? &wrs[i + 1] : nullptr;
            wrs[i].sg_list = &sges[3 * i];
            wrs[i].num_sge = 3;
            sges[3 * i].length = geometry.header_size;
            sges[3 * i + 1].length = geometry.payload_slot;
            sges[3 * i + 2].addr = (uintptr_t)sink;
            sges[3 * i + 2].length = sink_len;
            sges[3 * i + 2].lkey = sink_mr->lkey;
        }

        out_of_buffer_path =
            "/sys/class/infiniband/" + device.rdma_name + "/ports/1/hw_counters/out_of_buffer";
        std::ifstream f(out_of_buffer_path);
        if (!(f >> out_of_buffer_base)) {
            WARN("{:s} is not readable; NIC drops will not be reported", out_of_buffer_path);
            out_of_buffer_path.clear();
        }

        header_buf->register_producer(unique_name);
        payload_buf->register_producer(unique_name);
    } catch (...) {
        release();
        throw;
    }
}

ibvUdpRecv::~ibvUdpRecv() {
    release();
}

void ibvUdpRecv::release() {
    if (flow && ibv_destroy_flow(flow))
        ERROR("ibv_destroy_flow failed: {:s}", strerror(errno));
    if (qp && ibv_destroy_qp(qp))
        ERROR("ibv_destroy_qp failed: {:s}", strerror(errno));
    if (cq && ibv_destroy_cq(ibv_cq_ex_to_cq(cq)))
        ERROR("ibv_destroy_cq failed: {:s}", strerror(errno));
    if (channel && ibv_destroy_comp_channel(channel))
        ERROR("ibv_destroy_comp_channel failed: {:s}", strerror(errno));
    if (sink_mr && ibv_dereg_mr(sink_mr))
        ERROR("ibv_dereg_mr failed for the sink: {:s}", strerror(errno));
    free(sink);
    flow = nullptr;
    qp = nullptr;
    cq = nullptr;
    channel = nullptr;
    sink_mr = nullptr;
    sink = nullptr;
}

void ibvUdpRecv::main_thread() {
    frameID post_id(header_buf);
    for (int i = 0; i < frames_in_flight; i++) {
        if (!post_frame(post_id))
            return;
        post_id++;
    }
    // Steer only now that work requests are posted, or the first packets are dropped.
    flow = ibv_create_flow(qp, reinterpret_cast<ibv_flow_attr*>(&flow_rule));
    if (!flow)
        FATAL_ERROR("ibv_create_flow failed on {:s}: {:s}{:s}", device.netdev, strerror(errno),
                    errno == EPERM ? " (flow steering needs CAP_NET_RAW: setcap cap_net_raw+ep "
                                     "on the kotekan binary, or run as root)"
                                   : "");
    INFO("Receiving UDP port {:d} on {:s} ({:s}), {:d} byte packets, {:d} per frame, {:d} frames "
         "posted, {:s} polling",
         ntohs(flow_rule.udp.val.dst_port), device.netdev, device.rdma_name, geometry.wire_len(),
         geometry.packets_per_frame, frames_in_flight, event_mode ? "event" : "busy");

    if (!out_of_buffer_path.empty())
        stats_thread = std::thread(&ibvUdpRecv::track_out_of_buffer, this);
    while (!stop_thread) {
        const int polled = poll_completions();
        while (frames_to_post > 0) {
            if (!post_frame(post_id))
                return;
            post_id++;
            frames_to_post--;
        }
        if (polled > 0 || !event_mode)
            continue;
        // Nothing arrived: arm the CQ, poll once more, then sleep on the completion channel
        // (see wait_for_cq_event).
        if (!cq_armed) {
            if (ibv_req_notify_cq(ibv_cq_ex_to_cq(cq), 0))
                FATAL_ERROR("ibv_req_notify_cq failed: {:s}", strerror(errno));
            cq_armed = true;
            continue;
        }
        if (wait_for_cq_event(channel, 100))
            cq_armed = false;
    }
    if (stats_thread.joinable())
        stats_thread.join();
}

bool ibvUdpRecv::post_frame(int frame_id) {
    uint8_t* headers = header_buf->wait_for_empty_frame(unique_name, frame_id);
    if (headers == nullptr)
        return false;
    uint8_t* payload = payload_buf->wait_for_empty_frame(unique_name, frame_id);
    if (payload == nullptr)
        return false;
    const uint32_t header_lkey = header_mrs.lkey(frame_id);
    const uint32_t payload_lkey = payload_mrs.lkey(frame_id);
    for (size_t i = 0; i < geometry.packets_per_frame; i++) {
        wrs[i].wr_id = next_wr_id++;
        sges[3 * i].addr = (uintptr_t)(headers + i * geometry.header_slot);
        sges[3 * i].lkey = header_lkey;
        sges[3 * i + 1].addr = (uintptr_t)(payload + i * geometry.payload_slot);
        sges[3 * i + 1].lkey = payload_lkey;
    }
    ibv_recv_wr* bad = nullptr;
    const int rc = ibv_post_recv(qp, &wrs[0], &bad);
    if (rc)
        FATAL_ERROR("ibv_post_recv failed for frame {:d}: {:s}", frame_id, strerror(rc));
    return true;
}

int ibvUdpRecv::poll_completions() {
    ibv_poll_cq_attr poll_attr;
    memset(&poll_attr, 0, sizeof(poll_attr));
    int rc = ibv_start_poll(cq, &poll_attr);
    if (rc == ENOENT)
        return 0;
    if (rc)
        FATAL_ERROR("ibv_start_poll failed: {:s}", strerror(rc));

    int polled = 0;
    uint64_t bytes = 0;
    int bad_length = 0;
    int csum_errors = 0;
    while (true) {
        if (cq->status != IBV_WC_SUCCESS) {
            if (stop_thread && cq->status == IBV_WC_WR_FLUSH_ERR)
                break;
            FATAL_ERROR("Receive on {:s} failed: {:s} (wr_id {:d})", device.netdev,
                        ibv_wc_status_str(cq->status), cq->wr_id);
        }
        if (cq->wr_id != expected_wr_id)
            FATAL_ERROR("Receive completion for packet {:d} arrived before packet {:d}", cq->wr_id,
                        expected_wr_id);
        expected_wr_id++;

        const uint32_t wire_len = ibv_wc_read_byte_len(cq);
        const bool csum_ok = ibv_wc_read_wc_flags(cq) & IBV_WC_IP_CSUM_OK;
        udp_slot_tail* tail =
            udp_slot_tail_of(header_buf->frames[fill_frame_id] + fill_count * geometry.header_slot,
                             geometry.header_slot);
        tail->wire_len = wire_len;
        tail->flags = csum_ok ? UDP_SLOT_CSUM_OK : 0;
        bytes += wire_len;
        bad_length += wire_len != geometry.wire_len();
        csum_errors += !csum_ok;

        polled++;
        if (++fill_count == geometry.packets_per_frame) {
            header_buf->mark_frame_full(unique_name, fill_frame_id);
            payload_buf->mark_frame_full(unique_name, fill_frame_id);
            fill_frame_id = (fill_frame_id + 1) % header_buf->num_frames;
            fill_count = 0;
            frames_to_post++;
            frames_counter.inc();
        }
        // Bound the batch so a completed frame is replaced before the next one is polled.
        if (polled == (int)geometry.packets_per_frame)
            break;
        rc = ibv_next_poll(cq);
        if (rc == ENOENT)
            break;
        if (rc)
            FATAL_ERROR("ibv_next_poll failed: {:s}", strerror(rc));
    }
    ibv_end_poll(cq);

    packets_counter.inc(polled);
    bytes_counter.inc(bytes);
    if (bad_length)
        bad_length_counter.inc(bad_length);
    if (csum_errors)
        csum_error_counter.inc(csum_errors);
    return polled;
}

void ibvUdpRecv::track_out_of_buffer() {
    while (!stop_thread) {
        std::ifstream f(out_of_buffer_path);
        uint64_t value;
        if (f >> value)
            out_of_buffer_gauge.set(value - out_of_buffer_base);
        for (int i = 0; i < 10 && !stop_thread; i++)
            usleep(100000);
    }
}
