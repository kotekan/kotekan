#ifndef CRS_HEADER_GEN_HPP
#define CRS_HEADER_GEN_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stddef.h> // for size_t
#include <stdint.h> // for uint8_t, uint16_t, uint32_t
#include <string>   // for string

/**
 * @class crsHeaderGen
 * @brief Produces the CRS/UDP header frames for the F-engine's output packets.
 *
 * The second consumer of crsWindowAlign's job buffer, beside the F-engine stage; it never
 * touches payload buffers. One header frame is emitted per job, in order and in lockstep with
 * the F-engine's spectrum frames, so ibvUdpSend consumes the pair by frame id. A spectrum of
 * fft_length/2 bins goes out as (fft_length/2) / bins_per_packet packets; packet p of every
 * window carries stream_id = p, so a bin's frequency follows from stream_id alone, and
 * seq_number = the job's window_index, the output time code.
 *
 * Headers are written the first time each ring frame is used (udpPacketGen pattern); later
 * passes store only the seq_number, one 8-byte write per packet. Buffers with @c zero_frames
 * would wipe the reused slots; leave it off.
 *
 * @par Buffers
 * @buffer job_buf        One fft_window_job per frame, from crsWindowAlign
 *         @buffer_format fft_window_job_size(fft_length / samples_per_packet) bytes
 *         @buffer_metadata none
 * @buffer out_header_buf Header slots, produced in lockstep with the F-engine's spectrum frames
 *         @buffer_format (fft_length/2) / bins_per_packet slots of header_slot bytes
 *         @buffer_metadata none
 *
 * @conf src_mac         String. Source MAC.
 * @conf dst_mac         String. Destination MAC.
 * @conf src_ip          String. Source IPv4 address.
 * @conf dst_ip          String. Destination IPv4 address.
 * @conf src_port        Int. Source UDP port.
 * @conf dst_port        Int. Destination UDP port; one port for both ADCs.
 * @conf adc_id          Int, default 0. source_id.link_id of every packet.
 * @conf crate_id        Int, default 0. source_id field.
 * @conf slot_id         Int, default 0. source_id field.
 * @conf fft_length      Int. Samples per window; fft_length/2 bins go out per job.
 * @conf bins_per_packet Int, default 8192. Quantized bins (bytes) per packet; must divide
 *                       fft_length/2.
 * @conf header_size     Int. On-wire header bytes per packet; 64 for CRS packets.
 * @conf header_slot     Int. Header slot stride; see udp_slot_geometry.
 *
 * @author Andre Renard
 */
class crsHeaderGen : public kotekan::Stage {
public:
    crsHeaderGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~crsHeaderGen() override = default;
    void main_thread() override;

private:
    Buffer* job_buf;
    Buffer* out_buf;

    size_t header_size;
    size_t header_slot;
    uint8_t src_mac[6];
    uint8_t dst_mac[6];
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint32_t adc_id, crate_id, slot_id;
    uint32_t bins_per_packet;
    uint32_t packets_per_frame;
};

#endif /* CRS_HEADER_GEN_HPP */
