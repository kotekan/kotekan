#ifndef GNSS_GPU_RECORD_ASSEMBLE_HPP
#define GNSS_GPU_RECORD_ASSEMBLE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssElemCal.hpp"
#include "restServer.hpp"

#include <complex>
#include <mutex>
#include <vector>

/**
 * @class GnssGpuRecordAssemble
 * @brief Host tail of the phase-F GPU tracking chain: gnssGpuChain frames -> tracker records.
 *
 * Consumes the cudaProcess output (control block + raw per-channel E/P/L correlations, layout
 * gnssGpuChain.hpp) and performs GnssChannelizedTracker's pass-2: cross-channel summation over
 * each PRN's covering mask, the carrier-NCO phase integration + derotation (phase continuity
 * state lives here; the slope f_nco = ctrim + ff rides in the control block), and the
 * gnssRecord.hpp record floats. Emits one rec_buf frame per record window with the window's
 * absolute sample in GnssChanMetadata -- byte-compatible with the CPU tracker's output, so the
 * combiner/broker/viewer are untouched.
 *
 * @conf in_buf   gnssGpuChain frames from the cudaProcess chain
 * @conf out_buf  tracker record frames (n_prn * record_floats * float)
 * @conf prns     PRN list (must match the cudaGnssTrack command's)
 * @conf sample_rate  (for the NCO dt; default 5e6)
 */
class GnssGpuRecordAssemble : public kotekan::Stage {
public:
    GnssGpuRecordAssemble(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~GnssGpuRecordAssemble() override;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    std::vector<int> _prns;
    double _sample_rate;

    /// WALL-CLOCK FALLBACK, LATCHED ONCE (see main_thread). When the producer supplies no
    /// frame0_utc the records have no absolute anchor, and the old fallback stamped
    /// `system_clock::now()` per record -- which on CHORD puts the four sub-records of a frame
    /// microseconds apart instead of 10.49 ms, because they are assembled back to back. That is
    /// not a small error: every CROSS-RECORD estimator in GnssCoherentCombiner works in UTC and
    /// derives its grid from the MINIMUM consecutive spacing, so a burst of near-equal stamps
    /// scrambles the record order inside the transform. Anchoring once and extrapolating by
    /// wstart keeps the same (host-clock) origin while making the grid exactly uniform.
    double _wall_anchor = 0.0; ///< now() - wstart/rate at the first unanchored frame; 0 = unset
    uint64_t _no_utc0_frames = 0; ///< frames stamped from the fallback (for the rate-limited warn)

    /// Element axis (CHORD). 0 = single-antenna airspy layout, byte-for-byte.
    int _n_elements = 0;
    /// Which antenna the record HEADER's correlation slots carry -- the broker's loop reference.
    int _reference_element = 0;
    /// SELF-CALIBRATED ELEMENT SUM (gnssElemCal.hpp; CHORD_GNSS_STATE 8.21.5). When enabled the
    /// header correlation slots carry the calibrated weighted MEAN over all elements instead of
    /// the bare reference element: same phase convention (reference-anchored), same "one
    /// element" scale, per-record SNR up ~sqrt(N_healthy) -- which is what makes the per-record
    /// carrier phase estimable from one instance (the phase-floor fix) and hands the broker's
    /// DLL/carrier loops the array gain for free. Until each PRN's cal is warm (~3 tau of
    /// updates) the header is the reference element, byte-identical to the historical output.
    bool _elem_sum = false;
    double _elem_sum_tau_s = 0.5;  ///< cal EMA time constant -- fast enough to follow the
                                   ///< inter-element fringe rotation as a satellite transits
    double _elem_sum_min_w = 0.02; ///< weight gate vs the strongest element: absent/unpowered
                                   ///< elements (EMA of pure noise) fall below and are excluded
    std::vector<gnss::ElemCal> _cal; ///< per PRN slot
    std::vector<uint8_t> _anchor_warned; ///< one WARN per PRN when the phase anchor moves off
                                         ///< the reference element (a one-time phase step
                                         ///< downstream); cleared on cal reset
    /// Scratch, [n_rows_spec][n_elem]: the per-antenna covering-mask sum, reused per PRN so the
    /// per-record path does not allocate.
    std::vector<std::complex<double>> _g_elem;

    // NCO state per PRN slot (pass-2's half of the carrier machinery).
    std::vector<double> _phi;
    std::vector<double> _phi_cyc;   ///< NCO phase, UNWRAPPED, in cycles (the export's time base;
                                    ///< _phi is the same phase wrapped for the rotation)
    std::vector<double> _phi_cmd_prev; ///< previous record's commanded phase (cycles)
    std::vector<uint8_t> _phi_cmd_ok;
    std::vector<double> _fcar_prev; ///< previous record's replica f_ref (to size the re-pin step)
    std::vector<uint8_t> _fcar_prev_ok;
    std::vector<std::complex<double>> _a_prev;
    std::vector<uint8_t> _a_prev_ok;
    std::vector<int64_t> _wstart_prev;

    /// Per-channel PROMPT-phase dump (chan_dump_prn / chan_dump_decim / chan_dump_path):
    /// DIAGNOSTIC (2026-07-21, L5 ADR-wander): the channel-width A/B showed the wander
    /// amplitude depends on the despread channel set (narrow 5-ch = 5-6x WORSE than the
    /// full 10) -> the mechanism lives in the per-channel phases the cross-channel sum
    /// normally hides. For the one listed PRN, every decim-th record writes one line per
    /// covering channel: "utc ch corr_re corr_im energy" (raw, pre-NCO-rotation -- the
    /// cross-channel RELATIVE phases are the observable). ~60 KB/s at 100 Hz x 10 ch.
    int _chan_dump_prn = -1;   ///< PRN number to dump (-1 = disabled)
    int _chan_dump_decim = 10; ///< dump every Nth record of that PRN
    long long _chan_dump_ctr = 0;
    FILE* _chan_dump = nullptr;

    /// PER-CHANNEL PROMPT SPECTRUM (task #32, docs/CHORD_JOINT_TRACKING.md P1). The general
    /// form of the chan_dump above: for EVERY PRN, accumulate the NCO-derotated, element-
    /// combined prompt per covering channel over a window, and serve it on
    /// `<unique_name>/get_spectrum?window=N`. A delay is a phase ramp across frequency, and this
    /// is the
    /// sufficient statistic for the fleet-level phase-slope delay fit in the broker --
    /// per-(PRN, channel) complex sums, ~4 kB per poll, never per-element data (the 30 Gbps
    /// full-CHORD trap). The derotation is the SAME `rot` the record's prompt gets, one
    /// common phase per record: it stops the residual-carrier winding across the window
    /// without touching the cross-channel RELATIVE phases, which are the observable.
    /// Enabled by the presence of `channel_ids` in the config (the generator wires the same
    /// per-GPU list the despread runs); absent -> fully inert, airspy/legacy byte-identical.
    ///
    /// ⚠️ WINDOWS ARE ADDRESSABLE AND HOP-QUANTISED (task #53, 2026-08-12). They used to be
    /// "whatever accumulated since your last GET", with a reset-on-read -- so the window was
    /// defined by WHEN THE BROKER'S REQUEST ARRIVED, and the broker polls 12 instances
    /// SEQUENTIALLY. The instances were therefore never summing the same records, and the
    /// broker could not repair it: re-polling a laggard returns a NEW SHORTER window, never
    /// the records it missed. There is no way to ask for the past.
    ///
    /// That misalignment is not cosmetic. Each instance's channels are a COMB spanning
    /// ~18.75 MHz (7 channels, stride 16), so the cross-instance phase relationship is a delay
    /// ramp -2*pi*f*tau, and the broker was absorbing the window offset into a FREE PHASE PER
    /// INSTANCE fitted from the data it then summed -- a self-reference that aligns noise and,
    /// when it fails, drops the whole chain to the quadrature fallback (gps_l5 measured
    /// align 0.143 with 9/12 satellites on `quad`). See task #52.
    ///
    /// Now: window index = floor(wstart / _spec_win_samples), derived from the F-engine sample
    /// clock, so every instance assigns a record to the SAME window with no negotiation. A ring
    /// of completed windows lets a laggard still be asked for the window its peers already
    /// returned. Reads are IDEMPOTENT -- no reset -- so a second poller is harmless.
    std::vector<int> _spec_freq_ids;             ///< [n_chan] F-engine freq_id per channel
    int64_t _spec_win_samples = 0;               ///< window length, SAMPLES (0 = legacy mode)
    /// One accumulated window. Slot for index i is _spec_ring[i % depth], so a window is
    /// evicted only when the ring wraps past it -- no bookkeeping list, and the slot's own
    /// `idx` is what says whether it still holds what you asked for.
    struct SpecWindow {
        int64_t idx = -1;                        ///< window index, or -1 for an unused slot
        int64_t w0 = -1, w1 = -1;                ///< wstart of the first/last record in it
        std::vector<double> re, im, energy;      ///< [n_prn * n_chan]
        std::vector<int> nrec;                   ///< [n_prn]
    };
    std::vector<SpecWindow> _spec_ring;          ///< depth from config; index -> idx % depth
    int64_t _spec_max_idx = -1;                  ///< newest index SEEN; complete windows are < this
    std::mutex _spec_mtx;                        ///< guards _spec_* between main_thread and REST
    std::vector<std::complex<double>> _spec_scratch; ///< [n_elem] per-channel cal-combine input
    /// Accumulate one record's channels into the window that owns `wstart`, opening/clearing
    /// the ring slot on a boundary crossing. Caller holds _spec_mtx.
    SpecWindow& spec_window_for(int64_t wstart);
    void spectrum_callback(kotekan::connectionInstance& conn);
};

#endif
