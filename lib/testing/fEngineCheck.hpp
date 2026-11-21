#ifndef FENGINE_CHECK_HPP
#define FENGINE_CHECK_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter
#include "udpPacket.hpp"         // for udp_slot_geometry

#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class fEngineCheck
 * @brief Checks the spectrum windows that fEngineSimulate and crsHeaderGen emit against a
 *        known input tone.
 *
 * One frame pair is one window of fft_length/2 quantized bins. Per packet p the CRS headers
 * must carry the cookie, stream_id == p, source_id.link_id == adc_id, a seq_number equal
 * across the frame and incrementing by 1 frame to frame (the first frame sets the base), and a
 * slot tail of header_size + payload_size wire bytes. Per bin, with real = (byte >> 4) - 8 and
 * imag = (byte & 0xf) - 8: the tone bin's real must be within tolerance of expected_real and
 * its |imag| at most tolerance, and every other bin's |real| and |imag| at most tolerance (a
 * zero bin is 0x88). A window failing any check is counted bad. At least one window must hit
 * expected_real exactly, so a wrong gain cannot pass inside the tolerance.
 *
 * With num_windows >= 0 the stage exits kotekan with TEST_PASSED when at most max_bad_windows
 * windows were bad and the exact tone value was seen, TEST_FAILED otherwise (udpPacketCheck
 * pattern).
 *
 * With wire_mode the frames instead hold arrival-order packets from an ibvUdpRecv, where the
 * output streams of several ADCs interleave and frames do not align to windows. Every packet
 * is then checked on its own: the tail (checksum flag included), the cookie, link_id <
 * num_adcs, stream_id < fft_length / 2 / payload_size, a per-ADC non-decreasing seq_number,
 * and its payload slice of the spectrum — the ADC's tone bin when this stream_id carries it,
 * zeros elsewhere. tone_bins and expected_reals give one entry per ADC; a bad packet counts
 * toward max_bad_windows, and num_windows counts complete windows of packets per ADC.
 *
 * @par Buffers
 * @buffer header_buf   Header slots from crsHeaderGen, consumed in lockstep with payload_buf
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Quantized spectrum frames from fEngineSimulate
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf header_size     Int. See udp_slot_geometry; 64 for CRS packets.
 * @conf header_slot     Int. See udp_slot_geometry.
 * @conf payload_size    Int. See udp_slot_geometry; bins_per_packet of the output packets.
 * @conf payload_slot    Int. See udp_slot_geometry.
 * @conf adc_id          Int, default 0. The source_id.link_id every packet must carry.
 * @conf tone_bin        Int. Output bin of the input tone: the generator's tone_bin *
 *                       fft_length / samples_per_packet.
 * @conf expected_real   Int. Quantized real part of the tone bin in a loss-free window.
 * @conf tolerance       Int, default 1. Allowed deviation of every bin component.
 * @conf num_windows     Int, default -1 (forever). Windows to check before exiting kotekan.
 * @conf max_bad_windows Int, default 0. With num_windows, bad windows (or, in wire_mode, bad
 *                       packets) allowed before TEST_FAILED.
 * @conf wire_mode       Bool, default false. Arrival-order per-packet checking behind an
 *                       ibvUdpRecv.
 * @conf num_adcs        Int, default 1. Wire mode: ADCs sharing the port.
 * @conf fft_length      Int. Wire mode: the F-engine's window length, for the packet count
 *                       per window.
 * @conf tone_bins       List of int. Wire mode: each ADC's tone bin.
 * @conf expected_reals  List of int. Wire mode: each ADC's expected quantized tone real part.
 *
 * @par Metrics
 * @metric kotekan_fengine_check_windows_total     Windows checked.
 * @metric kotekan_fengine_check_bad_windows_total Windows failing a header or bin check.
 *
 * @author Andre Renard
 */
class fEngineCheck : public kotekan::Stage {
public:
    fEngineCheck(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~fEngineCheck() override = default;
    void main_thread() override;

private:
    /// Checks one window; returns true when every header and bin check passed.
    bool check_window(const uint8_t* headers, const uint8_t* payload);
    /// Wire mode: checks one packet; returns true when it passed.
    bool check_packet(const uint8_t* header, const uint8_t* payload);

    Buffer* header_buf;
    Buffer* payload_buf;
    const udp_slot_geometry geometry;

    uint32_t adc_id;
    uint32_t tone_bin;
    int expected_real;
    int tolerance;
    int num_windows;
    int64_t max_bad_windows;
    bool wire_mode;
    uint32_t num_adcs;
    uint32_t packets_per_window = 0;
    std::vector<int> tone_bins;
    std::vector<int> expected_reals;

    bool have_seq = false;
    uint64_t expected_seq = 0;
    bool saw_exact = false;
    int64_t bad_windows = 0;
    std::vector<uint64_t> packets_seen;
    std::vector<uint64_t> last_window;
    std::vector<bool> saw_exact_adc;

    kotekan::prometheus::Counter& windows_counter;
    kotekan::prometheus::Counter& bad_windows_counter;
};

#endif /* FENGINE_CHECK_HPP */
