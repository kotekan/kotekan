#include "crsPacketGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "crsPacket.hpp"       // for crs_packet_header, crs_header_of, pack12, CRS_PACKET_COOKIE
#include "kotekanLogging.hpp"  // for INFO
#include "udpPacket.hpp"       // for fill_udp_headers, parse_mac, parse_ipv4, udp_slot_tail_of
#include "visUtil.hpp"         // for frameID

#include "fmt.hpp" // for format

#include <algorithm>  // for clamp
#include <functional> // for bind
#include <math.h>     // for cosf, lround, M_PI
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for invalid_argument
#include <string.h>   // for memset
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(crsPacketGen);

crsPacketGen::crsPacketGen(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&crsPacketGen::main_thread, this)),
    header_buf(get_buffer("header_buf")), payload_buf(get_buffer("payload_buf")),
    geometry(udp_slot_geometry::from_config(config, unique_name, header_buf, payload_buf)) {

    if (geometry.header_size < sizeof(packet_headers))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: header_size {:d} has no room for the {:d} byte CRS headers"),
                        unique_name, geometry.header_size, sizeof(packet_headers)));
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
    dst_port_base = config.get_default<uint16_t>(unique_name, "dst_port_base", 10000);
    num_adcs = config.get_default<uint32_t>(unique_name, "num_adcs", 2);
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 4096);
    start_seq = config.get_default<uint64_t>(unique_name, "start_seq", 0);
    crate_id = config.get_default<uint32_t>(unique_name, "crate_id", 0);
    slot_id = config.get_default<uint32_t>(unique_name, "slot_id", 0);
    link_id = config.get_default<uint32_t>(unique_name, "link_id", 0);
    sample_mean = config.get_default<int>(unique_name, "sample_mean", 2048);
    tone_bin = config.get_default<std::vector<int>>(unique_name, "tone_bin", {});
    tone_amplitude = config.get_default<std::vector<float>>(unique_name, "tone_amplitude", {});
    noise_rms = config.get_default<float>(unique_name, "noise_rms", 0.0f);
    seed = config.get_default<uint32_t>(unique_name, "seed", 42);
    drop_period = config.get_default<uint64_t>(unique_name, "drop_period", 0);
    drop_burst = config.get_default<uint64_t>(unique_name, "drop_burst", 1);
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);

    if (num_adcs == 0 || geometry.packets_per_frame % num_adcs != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: num_adcs {:d} must divide packets_per_frame {:d}"), unique_name,
                        num_adcs, geometry.packets_per_frame));
    if (samples_per_packet % 2 != 0 || geometry.payload_size != samples_per_packet * 3 / 2)
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: payload_size {:d} does not hold samples_per_packet {:d} 12-bit samples"),
            unique_name, geometry.payload_size, samples_per_packet));
    if (start_seq % samples_per_packet != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: start_seq {:d} is not a multiple of samples_per_packet {:d}"),
                        unique_name, start_seq, samples_per_packet));
    if (!tone_bin.empty() && (tone_bin.size() != num_adcs || tone_amplitude.size() != num_adcs))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: tone_bin and tone_amplitude need one entry per ADC ({:d})"),
                        unique_name, num_adcs));

    header_buf->register_producer(unique_name);
    payload_buf->register_producer(unique_name);
}

void crsPacketGen::main_thread() {
    frameID frame_id(header_buf);
    std::vector<bool> initialized(header_buf->num_frames, false);
    const uint16_t udp_payload_len =
        geometry.header_size - sizeof(udp_packet_headers) + geometry.payload_size;

    // One period of each ADC's tone, around sample_mean; packets of an ADC are identical.
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, noise_rms);
    std::vector<std::vector<float>> tone(num_adcs,
                                         std::vector<float>(samples_per_packet, sample_mean));
    for (uint32_t a = 0; a < num_adcs && !tone_bin.empty(); a++)
        for (uint32_t k = 0; k < samples_per_packet; k++)
            tone[a][k] +=
                tone_amplitude[a] * cosf(2.0f * M_PI * tone_bin[a] * k / samples_per_packet);

    std::vector<uint64_t> seq(num_adcs, start_seq);
    std::vector<uint64_t> packets(num_adcs, 0);
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
                const uint32_t adc = j % num_adcs;
                uint8_t* slot = headers + j * geometry.header_slot;
                memset(slot, 0, geometry.header_slot);
                fill_udp_headers(reinterpret_cast<udp_packet_headers*>(slot), src_mac, dst_mac,
                                 src_ip, dst_ip, src_port, dst_port_base + adc, udp_payload_len);
                crs_packet_header* crs = crs_header_of(slot);
                crs->cookie = CRS_PACKET_COOKIE;
                crs->source_id.link_id = link_id;
                crs->source_id.slot_id = slot_id;
                crs->source_id.crate_id = crate_id;
                crs->stream_id = adc;
                udp_slot_tail* tail = udp_slot_tail_of(slot, geometry.header_slot);
                tail->wire_len = geometry.wire_len();
                tail->flags = UDP_SLOT_CSUM_OK;
                uint8_t* data = payload + j * geometry.payload_slot;
                for (uint32_t k = 0; k < samples_per_packet; k++) {
                    const long v = lround(tone[adc][k] + (noise_rms > 0 ? noise(rng) : 0.0f));
                    pack12(data, k, (uint16_t)std::clamp(v, 0l, 4095l));
                }
            }
            initialized[frame_id] = true;
        }
        for (size_t j = 0; j < geometry.packets_per_frame; j++) {
            const uint32_t adc = j % num_adcs;
            if (drop_period > 0 && ++packets[adc] % drop_period == 0)
                seq[adc] += drop_burst * samples_per_packet;
            crs_header_of(headers + j * geometry.header_slot)->seq_number = seq[adc];
            seq[adc] += samples_per_packet;
        }

        header_buf->mark_frame_full(unique_name, frame_id);
        payload_buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
        produced++;
    }
    INFO("Produced {:d} frames of {:d} packets, {:d} ADCs ending at seq {:d}", produced,
         geometry.packets_per_frame, num_adcs, seq[0]);
}
