/**
 * @file
 * @brief Tap the CHORD F-engine voltage buffer for the GNSS chain, drop-tolerantly.
 *  - GnssChordVoltageTap : public kotekan::Stage
 */

#ifndef GNSS_CHORD_VOLTAGE_TAP_HPP
#define GNSS_CHORD_VOLTAGE_TAP_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <atomic> // for atomic (live element selection)
#include <mutex>  // for mutex (element_power snapshot)
#include <string>
#include <vector>

/**
 * @class GnssChordVoltageTap
 * @brief Extract the GNSS covering channels from CHORD's `host_voltage_buffer`.
 *
 * The entry point of the CHORD GNSS chain. Reads the F-engine voltage in its native
 * `[time][freq][pol][dish]` ndarray layout and copies out just the covering channels for the
 * signal being tracked, as `[hop][chan][element]` -- which is exactly what
 * @ref gnss_cuda::launch_correlate_nm consumes, so nothing downstream transposes.
 *
 * WHY EXTRACT ON THE CPU RATHER THAN UPLOAD AND SELECT. A full frame is 8192 hops x 384
 * channels x 128 elements = 402 MB every 41.94 ms, i.e. 9.6 GB/s of PCIe per GPU. The GNSS
 * signal occupies ~14 of those 384 channels, so uploading the frame to select 3.6% of it would
 * spend the entire record period on the transfer. Extracting first ships ~15 MB instead.
 *
 * THIS STAGE IS THE VALVE, and that is not a convenience. It taps a buffer that the production
 * pipeline produces into: registering as a consumer means the frame is not released until
 * EVERY consumer marks it empty, so a stall anywhere in the GNSS branch would back-pressure the
 * F-engine ingest of the science pipeline itself. So when the output is full this stage drops
 * the input frame immediately and counts it, exactly as @ref Valve does -- and it does the drop
 * itself rather than feeding a separate Valve, because a Valve placed upstream would first copy
 * all 402 MB and one placed downstream would leave this stage free to block.
 *
 * `kotekan_gnss_tap_dropped_frames_total` is therefore the silent-data-loss observable for the
 * GNSS branch, the same role `valve_dropped_frames_total` plays on the airspy node: a dropped
 * frame becomes a gap that the despread ring zero-fills, which costs SNR in any coherent window
 * touching it and looks exactly like decohered signal if you are not watching the counter.
 *
 * TIME. CHORD's `fpga_seq_num` counts 5.12 us PFB frames (hops), while the GNSS chain anchors
 * replicas to an absolute PRE-CHANNELIZATION sample index. The conversion is one multiply by
 * `fft_length` (16384), done here so that everything downstream keeps the airspy chain's
 * meaning of @c sample_seq untouched.
 *
 * @conf in_buf        CHORD voltage ndarray [time][freq][pol][dish], 4+4b
 * @conf out_buf       [hop][chan][element] bytes
 * @conf chan_ids      frame channel indices to extract (the covering set; see chord_band_plan.py)
 * @conf n_elements    elements to extract
 * @conf element_offset first element (default 0)
 * @conf frame_chan_stride  channel axis stride of the input frame (num_local_freq)
 * @conf frame_elem_stride  element axis stride of the input frame (num_elements)
 * @conf samples_per_data_set hops per frame
 * @conf fft_length    real samples per hop (16384 on CHORD) -- the seq -> sample conversion
 * @conf band_power_chans  local channel indices to monitor for power/clip (empty = OFF, #8)
 * @conf band_power_period_s  seconds between passes (default 10)
 * @conf band_power_hop_stride  sample every Nth hop within a pass (default 32)
 */
class GnssChordVoltageTap : public kotekan::Stage {
public:
    GnssChordVoltageTap(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~GnssChordVoltageTap() override = default;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    std::vector<int> _chan_ids;
    int _n_elements;
    /// LIVE-SETTABLE via POST <unique_name>/set_element {"element": N}. Antennas come and go
    /// (2026-08-07: element 0, the search's reference, was dark and the whole acquisition chain
    /// read zeros for a session), and re-picking one should not need a regenerate + a six-node
    /// restart. Atomic because the REST thread writes it while main_thread reads it per frame;
    /// a torn read is impossible for an int, and a frame straddling a change is harmless -- the
    /// consumer re-derives everything per frame.
    std::atomic<int> _element_offset;

    /// Per-element mean power from the last frame, for GET <unique_name>/element_power.
    /// Indexed by ABSOLUTE element (0.._frame_elem_stride), so it answers "where should the tap
    /// point?" rather than describing only what it currently taps.
    std::mutex _pow_lock;
    std::vector<double> _elem_power;
    long _pow_frames = 0;
    static constexpr int _pow_stride = 32; ///< sample every Nth hop (0.2% of the frame)

    // ── #8: RF-PATH HEALTH -- clip fraction + band power ────────────────────────────────
    /// OFF unless `band_power_chans` is non-empty: the pass never runs, /rf_stats reports
    /// enabled=false, and this stage behaves byte-identically to before. The generator arms
    /// it, so the C++ is inert until a config says otherwise.
    ///
    /// WHY THIS LIVES HERE AND IS NOT A NEW STAGE. This stage already IS the valve (see the
    /// class note above), already holds the full input frame, already decodes 4+4b, and
    /// already drops rather than blocks. A second consumer on `host_voltage_buffer` would add
    /// a second thing that must mark frames empty promptly -- another way to back-pressure the
    /// F-engine ingest -- to measure bytes this stage is holding anyway.
    ///
    /// ⚠️ CLIP IS PER (CHANNEL, ELEMENT), NOT PER ELEMENT. The 4+4b quantisation happens
    /// AFTER the PFB, so every channel rails independently and a narrowband source rails only
    /// its own. That per-channel signature is the entire point (#56, and the 08-18 1176 MHz
    /// event was band-selective at +16 dB); a per-element-only number averages it away.
    std::mutex _rf_lock;
    std::vector<int> _bp_chans;         ///< local channel indices to monitor (empty = OFF)
    std::vector<double> _bp_power;      ///< per _bp_chans: mean |x|^2 over elements and hops
    std::vector<double> _bp_clip_lo;    ///< per _bp_chans: fraction of nibbles at -8
    std::vector<double> _bp_clip_hi;    ///< per _bp_chans: fraction of nibbles at +7
    std::vector<double> _bp_elem_pow;   ///< per ABSOLUTE element, over _bp_chans
    std::vector<double> _bp_elem_clip;  ///< per ABSOLUTE element, (lo+hi) fraction
    double _bp_period_s;                ///< seconds between passes
    int _bp_hop_stride;                 ///< sample every Nth hop within a pass
    double _bp_last_s = 0.0;            ///< current_time() of the last pass; 0 = never run
    double _bp_cost_ms = 0.0;           ///< wall cost of the last pass -- SERVED, not assumed
    long _bp_passes = 0;
    int64_t _bp_seq = -1;               ///< fpga_seq_num of the frame the last pass measured
    int _frame_chan_stride;
    int _frame_elem_stride;
    int _n_hops;
    int _fft_length;
};

#endif // GNSS_CHORD_VOLTAGE_TAP_HPP
