#ifndef CRS_WINDOW_ALIGN_HPP
#define CRS_WINDOW_ALIGN_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "fftWindowJob.hpp"      // for fft_window_job
#include "prometheusMetrics.hpp" // for Counter, Gauge
#include "udpPacket.hpp"         // for udp_slot_geometry

#include <stdint.h> // for uint8_t, uint32_t, uint64_t
#include <string>   // for string

/**
 * @class crsWindowAlign
 * @brief Assembles FFT windows from received CRS packets: reads the header buffer and emits
 *        one fft_window_job per fft_length samples, aligned to seq_number % fft_length == 0.
 *
 * ibvUdpRecv stores packets in arrival order and never looks at them; this stage reads each
 * header slot's CRS seq_number and builds the window's lookup table of payload slot
 * addresses in time order (see fftWindowJob.hpp for the job layout, the consumer protocol
 * and the payload-frame cursor rules). Payload data is never touched or copied: the stage
 * consumes only header frames, and takes the payload buffer's frame base addresses without
 * registering as its consumer — the FFT stage is the payload consumer and holds those frames
 * until its window is done.
 *
 * Samples before the first multiple of fft_length are discarded. A missing packet points its
 * lookup entry at a noise-buffer slot chosen by a PRNG (no periodicity, so heavy loss adds
 * no comb lines). A gap that would synthesize more than max_gap_windows whole windows of
 * pure noise abandons the current window and re-aligns at the next boundary instead, so a
 * D-engine pause does not fabricate unbounded noise. A seq_number a full window older than
 * the current window start is a FATAL_ERROR: the D-engine restarted, and the daemon should
 * restart this side too.
 *
 * The stage fills the noise buffer's single frame once at startup with 12-bit-packed
 * Gaussian noise and holds backpressure order: when the FFT stage falls behind, job frames
 * run out, this stage blocks, header frames fill, and ibvUdpRecv stops posting — the NIC
 * then drops packets, which return as seq gaps and noise fill. The pipeline recovers by
 * itself; nothing deadlocks.
 *
 * @par Buffers
 * @buffer header_buf   Header slots from ibvUdpRecv
 *         @buffer_format packets_per_frame slots of header_slot bytes
 *         @buffer_metadata none
 * @buffer payload_buf  Payload slots from ibvUdpRecv; only frame addresses are read
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 * @buffer job_buf      One fft_window_job + lookup table per frame
 *         @buffer_format fft_window_job_size(fft_length / samples_per_packet) bytes
 *         @buffer_metadata none
 * @buffer noise_buf    One frame of noise_slots payload-sized 12-bit noise slots, filled once
 *         @buffer_format noise_slots slots of payload_slot bytes
 *         @buffer_metadata none
 *
 * @conf fft_length         Int. Samples per window; a multiple of samples_per_packet.
 * @conf samples_per_packet Int, default 4096. 12-bit samples per packet.
 * @conf adc_id             Int, default 0. The stream_id every packet must carry.
 * @conf header_size        Int. See udp_slot_geometry; 64 for CRS packets.
 * @conf header_slot        Int. See udp_slot_geometry.
 * @conf payload_size       Int. See udp_slot_geometry.
 * @conf payload_slot       Int. See udp_slot_geometry.
 * @conf queue_depth        Int, default 8192. ibvUdpRecv's queue depth, for the sizing check:
 *                          payload frames must cover one window plus the receive stage's
 *                          posted frames plus slack.
 * @conf noise_slots        Int, default 1024. Slots in the noise buffer.
 * @conf noise_mean         Float, default 0. Mean of the noise samples, in ADC counts.
 * @conf noise_rms          Float, default 100. RMS of the noise samples, in ADC counts.
 * @conf seed               Int, default 42. Noise fill and slot-choice PRNG seed.
 * @conf max_gap_windows    Int, default 1. Whole windows of pure noise to synthesize inside
 *                          one gap before abandoning and re-aligning.
 * @conf stall_warn_s       Float, default 5. WARN when no header frame arrives for this long.
 * @conf stall_fatal_s      Float, default 0 (off). FATAL_ERROR when no header frame arrives
 *                          for this long.
 *
 * @par Metrics
 * @metric kotekan_crs_window_align_packets_total     Header slots scanned.
 * @metric kotekan_crs_window_align_bad_packets_total Slots failing the tail, cookie,
 *                                                    stream_id or seq alignment checks.
 * @metric kotekan_crs_window_align_discarded_packets_total Packets before the alignment
 *                                                    boundary, at startup or after a realign.
 * @metric kotekan_crs_window_align_reordered_packets_total Packets whose seq was already
 *                                                    passed within the current window.
 * @metric kotekan_crs_window_align_lost_packets_total Lookup entries filled with noise.
 * @metric kotekan_crs_window_align_windows_total     Jobs emitted.
 * @metric kotekan_crs_window_align_realigns_total    Re-alignments after over-long gaps.
 * @metric kotekan_crs_window_align_current_seq       Newest good seq_number seen.
 *
 * @author Andre Renard
 */
class crsWindowAlign : public kotekan::Stage {
public:
    crsWindowAlign(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~crsWindowAlign() override = default;
    void main_thread() override;

private:
    /// Fills the noise buffer's one frame with packed Gaussian noise. False on shutdown.
    bool fill_noise();
    /// Points the current job at the next empty job frame. False on shutdown.
    bool ensure_job();
    /// Handles one good packet at header frame @p abs_frame, slot @p slot. False on shutdown.
    bool process_packet(uint64_t seq, uint64_t abs_frame, size_t slot);
    /// Appends one noise entry to the lookup table. False on shutdown (via emit_window).
    bool fill_noise_entry();
    /// Finalizes and emits the current window. @p closer_frame is the frame of the packet
    /// being processed; @p closed_by_packet when that packet is the window's last sample.
    /// False on shutdown.
    bool emit_window(uint64_t closer_frame, size_t closer_slot, bool closed_by_packet);
    /// Abandons the current window and re-aligns at the window containing @p seq or later.
    void realign(uint64_t seq);

    Buffer* header_buf;
    Buffer* payload_buf;
    Buffer* job_buf;
    Buffer* noise_buf;
    const udp_slot_geometry geometry;

    uint64_t fft_length;
    uint32_t samples_per_packet;
    uint32_t num_packets;
    uint32_t adc_id;
    uint32_t noise_slots;
    float noise_mean;
    float noise_rms;
    uint32_t seed;
    uint32_t max_gap_windows;
    float stall_warn_s;
    float stall_fatal_s;

    /// Alignment and window state.
    bool aligned = false;
    uint64_t window_start = 0;
    uint64_t expected_seq = 0;
    uint32_t cursor = 0;
    uint32_t lost_in_window = 0;
    uint64_t last_ref_frame = 0;
    bool have_ref = false;
    uint32_t noise_window_run = 0;
    bool realign_pending = false;
    uint64_t prev_acquire = 0;
    uint64_t prev_release = 0;

    /// The job frame being filled.
    fft_window_job* job = nullptr;
    uint64_t* lut = nullptr;
    int job_frame_id = 0;

    /// Scan position.
    uint64_t abs_frame = 0;

    uint8_t* noise_base = nullptr;
    uint64_t prng_state;

    kotekan::prometheus::Counter& packets_counter;
    kotekan::prometheus::Counter& bad_counter;
    kotekan::prometheus::Counter& discarded_counter;
    kotekan::prometheus::Counter& reordered_counter;
    kotekan::prometheus::Counter& lost_counter;
    kotekan::prometheus::Counter& windows_counter;
    kotekan::prometheus::Counter& realigns_counter;
    kotekan::prometheus::Gauge& current_seq_gauge;
};

#endif /* CRS_WINDOW_ALIGN_HPP */
