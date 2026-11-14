#include "crsHeaderGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "crsPacket.hpp"       // for crs_packet_header, crs_header_of, CRS_PACKET_COOKIE
#include "fftWindowJob.hpp"    // for fft_window_job
#include "udpPacket.hpp"       // for fill_udp_headers, parse_mac, parse_ipv4, udp_slot_tail_of
#include "visUtil.hpp"         // for frameID

#include "fmt.hpp" // for format

#include <functional> // for bind
#include <stdexcept>  // for invalid_argument
#include <string.h>   // for memset
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(crsHeaderGen);

crsHeaderGen::crsHeaderGen(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&crsHeaderGen::main_thread, this)),
    job_buf(get_buffer("job_buf")), out_buf(get_buffer("out_header_buf")) {

    header_size = config.get<size_t>(unique_name, "header_size");
    header_slot = config.get_default<size_t>(unique_name, "header_slot",
                                             udp_slot_geometry::default_header_slot(header_size));
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
    adc_id = config.get_default<uint32_t>(unique_name, "adc_id", 0);
    crate_id = config.get_default<uint32_t>(unique_name, "crate_id", 0);
    slot_id = config.get_default<uint32_t>(unique_name, "slot_id", 0);
    const uint32_t fft_length = config.get<uint32_t>(unique_name, "fft_length");
    bins_per_packet = config.get_default<uint32_t>(unique_name, "bins_per_packet", 8192);

    if (header_size < sizeof(packet_headers) || header_slot < header_size + sizeof(udp_slot_tail))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: header_slot {:d} cannot hold the {:d} byte CRS headers plus the tail"),
            unique_name, header_slot, sizeof(packet_headers)));
    if (bins_per_packet == 0 || (fft_length / 2) % bins_per_packet != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: bins_per_packet {:d} does not divide fft_length/2 = {:d}"),
                        unique_name, bins_per_packet, fft_length / 2));
    packets_per_frame = (fft_length / 2) / bins_per_packet;
    // stream_id numbers a window's packets in 16 bits.
    if (packets_per_frame > 65536)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: a window of {:d} packets overflows the 16-bit stream_id; "
                            "fft_length / 2 / bins_per_packet must be at most 65536"),
                        unique_name, packets_per_frame));
    if (out_buf->frame_size != packets_per_frame * header_slot)
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: out_header_buf frames hold {:d} bytes but {:d} packets of {:d} byte "
                "slots need {:d}"),
            unique_name, out_buf->frame_size, packets_per_frame, header_slot,
            packets_per_frame * header_slot));
    if (job_buf->frame_size < sizeof(fft_window_job))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: job_buf frames hold {:d} bytes but a job needs {:d}"),
                        unique_name, job_buf->frame_size, sizeof(fft_window_job)));

    job_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

void crsHeaderGen::main_thread() {
    frameID job_id(job_buf), out_id(out_buf);
    std::vector<bool> initialized(out_buf->num_frames, false);
    const uint16_t udp_payload_len = header_size - sizeof(udp_packet_headers) + bins_per_packet;

    while (!stop_thread) {
        const uint8_t* job_frame = job_buf->wait_for_full_frame(unique_name, job_id);
        if (job_frame == nullptr)
            break;
        const fft_window_job* job = reinterpret_cast<const fft_window_job*>(job_frame);
        uint8_t* headers = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (headers == nullptr)
            break;

        if (!initialized[out_id]) {
            for (uint32_t p = 0; p < packets_per_frame; p++) {
                uint8_t* slot = headers + p * header_slot;
                memset(slot, 0, header_slot);
                fill_udp_headers(reinterpret_cast<udp_packet_headers*>(slot), src_mac, dst_mac,
                                 src_ip, dst_ip, src_port, dst_port, udp_payload_len);
                crs_packet_header* crs = crs_header_of(slot);
                crs->cookie = CRS_PACKET_COOKIE;
                crs->source_id.link_id = adc_id;
                crs->source_id.slot_id = slot_id;
                crs->source_id.crate_id = crate_id;
                crs->stream_id = p;
                udp_slot_tail* tail = udp_slot_tail_of(slot, header_slot);
                tail->wire_len = header_size + bins_per_packet;
                tail->flags = UDP_SLOT_CSUM_OK;
            }
            initialized[out_id] = true;
        }
        for (uint32_t p = 0; p < packets_per_frame; p++)
            crs_header_of(headers + p * header_slot)->seq_number = job->window_index;

        job_buf->mark_frame_empty(unique_name, job_id);
        job_id++;
        out_buf->mark_frame_full(unique_name, out_id);
        out_id++;
    }
}
