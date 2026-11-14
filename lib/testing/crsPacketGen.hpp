#ifndef CRS_PACKET_GEN_HPP
#define CRS_PACKET_GEN_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "udpPacket.hpp"       // for udp_slot_geometry

#include <stdint.h> // for uint8_t, uint16_t, uint32_t, uint64_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class crsPacketGen
 * @brief Simulates a D-engine: fills a header buffer and a payload buffer with CRS packets
 *        of 12-bit ADC samples for ibvUdpSend.
 *
 * Slot j of every frame carries ADC `j % num_adcs`: its packets get `stream_id` = ADC number,
 * UDP destination port `dst_port_base` + ADC number, and a seq_number counting that ADC's
 * samples (+samples_per_packet per packet). Payloads are 12-bit unsigned samples (pack12): a
 * per-ADC tone of `tone_bin` cycles per packet plus optional Gaussian noise around
 * `sample_mean`. Headers and payloads are written the first time each ring frame is used;
 * later passes store only the seq_number, one 8-byte write per packet, so the stage cannot be
 * the bottleneck at line rate (udpPacketGen pattern). The tone is exactly periodic per packet,
 * so the reused payloads stay phase-coherent for any FFT window; the noise repeats with the
 * ring period, which a simulator can live with. Buffers with @c zero_frames would wipe the
 * reused slots; leave it off.
 *
 * A tone of `tone_bin` cycles per packet lands exactly on output bin
 * `tone_bin * fft_length / samples_per_packet` of the F-engine.
 *
 * `drop_period` injects packet loss: every drop_period'th packet of an ADC advances that ADC's
 * seq_number by `drop_burst` extra packets. The receiver sees clean seq gaps; nothing is held
 * back on the wire.
 *
 * @par Buffers
 * @buffer header_buf   Header slots, produced in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf src_mac            String. Source MAC.
 * @conf dst_mac            String. Destination MAC.
 * @conf src_ip             String. Source IPv4 address.
 * @conf dst_ip             String. Destination IPv4 address.
 * @conf src_port           Int. Source UDP port.
 * @conf dst_port_base      Int, default 10000. Destination port of ADC a is dst_port_base + a.
 * @conf num_adcs           Int, default 2. Must divide packets_per_frame.
 * @conf samples_per_packet Int, default 4096. Even; payload_size must be 3/2 of it.
 * @conf start_seq          Int, default 0. First seq_number of every ADC, in samples; a multiple
 *                          of samples_per_packet. Pick one that is not a multiple of the
 *                          F-engine's fft_length to exercise its startup discard.
 * @conf crate_id           Int, default 0. source_id field.
 * @conf slot_id            Int, default 0. source_id field.
 * @conf link_id            Int, default 0. source_id field.
 * @conf sample_mean        Int, default 2048. Mid-scale of the 12-bit samples.
 * @conf tone_bin           List of int, default empty (no tone). Per-ADC tone, in cycles per
 *                          packet; num_adcs entries.
 * @conf tone_amplitude     List of float, default empty. Per-ADC tone amplitude in ADC counts;
 *                          num_adcs entries when tone_bin is given.
 * @conf noise_rms          Float, default 0. Gaussian noise in ADC counts, drawn once per slot.
 * @conf seed               Int, default 42. Noise RNG seed.
 * @conf drop_period        Int, default 0 (off). Inject a seq gap every this many packets per ADC.
 * @conf drop_burst         Int, default 1. Packets skipped per injected gap.
 * @conf num_frames         Int, default -1 (unlimited). Frames to produce before the stage exits.
 * @conf header_size        Int. See udp_slot_geometry; 64 for CRS packets.
 * @conf header_slot        Int. See udp_slot_geometry.
 * @conf payload_size       Int. See udp_slot_geometry.
 * @conf payload_slot       Int. See udp_slot_geometry.
 *
 * @author Andre Renard
 */
class crsPacketGen : public kotekan::Stage {
public:
    crsPacketGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~crsPacketGen() override = default;
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
    uint16_t dst_port_base;
    uint32_t num_adcs;
    uint32_t samples_per_packet;
    uint64_t start_seq;
    uint32_t crate_id, slot_id, link_id;
    int sample_mean;
    std::vector<int> tone_bin;
    std::vector<float> tone_amplitude;
    float noise_rms;
    uint32_t seed;
    uint64_t drop_period;
    uint64_t drop_burst;
    int num_frames;
};

#endif /* CRS_PACKET_GEN_HPP */
