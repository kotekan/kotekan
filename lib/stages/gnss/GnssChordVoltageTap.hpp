/**
 * @file
 * @brief Tap the CHORD F-engine voltage buffer for the GNSS chain, dropping rather than blocking.
 *  - GnssChordVoltageTap : public kotekan::Stage
 */

#ifndef GNSS_CHORD_VOLTAGE_TAP_HPP
#define GNSS_CHORD_VOLTAGE_TAP_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <atomic> // for atomic (live element selection)
#include <mutex>  // for mutex (element_power and rf_stats snapshots)
#include <string>
#include <vector>

/**
 * @class GnssChordVoltageTap
 * @brief Extract the GNSS covering channels from CHORD's `host_voltage_buffer`.
 *
 * Reads the F-engine voltage in its native `[time][freq][pol][dish]` layout and copies out the
 * channels in @c chan_ids for @c n_elements elements, as `[hop][chan][element]` 4+4b bytes: the
 * frame's own axis order, so no consumer has to transpose.
 *
 * The extraction runs on the CPU because uploading the whole frame to select a few channels
 * would cost 9.6 GB/s of PCIe per GPU: a full frame is 8192 hops x 384 channels x 128 elements
 * = 402 MB every 41.94 ms, and a signal's ~14 covering channels are 3.6% of it, ~15 MB.
 *
 * The stage is its own valve. in_buf belongs to the science pipeline and a frame is released
 * only when every consumer has marked it empty, so a stall in the GNSS branch would
 * back-pressure the F-engine ingest. When the output is full this stage drops the input frame
 * at once, as @ref Valve does. It drops itself rather than feeding a Valve because a Valve
 * upstream would first copy the whole frame, and one downstream would leave this stage free to
 * block. Drops show only in `kotekan_gnss_tap_dropped_frames_total` and a rate-limited warning;
 * downstream, each one is a gap in sample_seq.
 *
 * CHORD's `fpga_seq_num` counts 5.12 us PFB frames (hops), while the GNSS chain anchors
 * replicas to an absolute pre-channelization sample index. The output's
 * GnssChanMetadata::sample_seq is fpga_seq_num * @c fft_length.
 *
 * REST: POST unique_name/set_element re-points the tapped elements; GET
 * unique_name/element_power and GET unique_name/rf_stats serve per-element power and the
 * RF-path monitor.
 *
 * @conf in_buf        CHORD voltage ndarray [time][freq][pol][dish], 4+4b
 * @conf out_buf       [hop][chan][element] bytes
 * @conf chan_ids      frame channel indices to extract (covering set; config/chord_band_plan.py)
 * @conf n_elements    elements to extract
 * @conf element_offset first element (default 0)
 * @conf frame_chan_stride  channel axis stride of the input frame (num_local_freq)
 * @conf frame_elem_stride  element axis stride of the input frame (num_elements)
 * @conf samples_per_data_set hops per frame
 * @conf fft_length    real samples per hop (default 16384), for the seq -> sample conversion
 * @conf band_power_chans  local channel indices to monitor for power and clipping (empty = off)
 * @conf band_power_period_s  seconds between passes (default 10)
 * @conf band_power_hop_stride  sample every Nth hop within a pass (default 32)
 *
 * @metric kotekan_gnss_tap_dropped_frames_total  input frames dropped because out_buf was full
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
    /// First tapped element, settable at runtime via POST <unique_name>/set_element
    /// {"element": N} so a dark element can be replaced without regenerating configs. Atomic
    /// because the REST thread writes it while main_thread reads it once per frame.
    std::atomic<int> _element_offset;

    /// Per-element mean power of the last copied frame, for GET <unique_name>/element_power.
    /// Indexed by absolute element [0, _frame_elem_stride), not by tapped element, so it shows
    /// where the tap could point.
    std::mutex _pow_lock;
    std::vector<double> _elem_power;
    long _pow_frames = 0;
    static constexpr int _pow_stride = 32; ///< element_power samples every Nth hop

    // ---- RF-path health: clip fraction and band power ----
    /// Off unless `band_power_chans` is non-empty; off means the pass never runs and rf_stats
    /// reports enabled=false. It lives in this stage, not a new one, because a second consumer
    /// of host_voltage_buffer would be one more stage that must release frames promptly to
    /// avoid back-pressuring the ingest, only to read bytes this stage already holds.
    ///
    /// Clipping is reported per channel as well as per element: the 4+4b quantization follows
    /// the PFB, so each channel rails independently and a narrowband source rails only its own
    /// channels, which a per-element figure averages away.
    std::mutex _rf_lock;
    std::vector<int> _bp_chans; ///< local channel indices to monitor (empty = off)
    /// Per _bp_chans: the absolute freq_id, read from the frame's chord metadata (coarse_freq),
    /// not from config; empty until a pass has read them from a chord-metadata frame. A local comb
    /// position means nothing off the node; freq_id x 0.1953125 MHz names the frequency.
    std::vector<int> _bp_freq_ids;
    std::vector<double> _bp_power;     ///< per _bp_chans: mean |x|^2 over elements and hops
    std::vector<double> _bp_clip_lo;   ///< per _bp_chans: fraction of nibbles at -8
    std::vector<double> _bp_clip_hi;   ///< per _bp_chans: fraction of nibbles at +7
    std::vector<double> _bp_elem_pow;  ///< per absolute element, mean |x|^2 over _bp_chans
    std::vector<double> _bp_elem_clip; ///< per absolute element, fraction of nibbles at -8 or +7
    double _bp_period_s;               ///< seconds between passes
    int _bp_hop_stride;                ///< sample every Nth hop within a pass
    double _bp_last_s = 0.0;           ///< current_time() of the last pass; 0 = never run
    double _bp_cost_ms = 0.0;          ///< wall time of the last pass, served as rf_stats cost_ms
    long _bp_passes = 0;
    int64_t _bp_seq = -1; ///< fpga_seq_num of the frame the last pass measured
    int _frame_chan_stride;
    int _frame_elem_stride;
    int _n_hops;
    int _fft_length;
};

#endif // GNSS_CHORD_VOLTAGE_TAP_HPP
