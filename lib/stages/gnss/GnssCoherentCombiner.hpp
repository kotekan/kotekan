/**
 * @file
 * @brief Coherently recombine per-subband GNSS despread products into the full-band complex
 *        amplitude.
 *  - GnssCoherentCombiner : public kotekan::Stage
 */

#ifndef GNSS_COHERENT_COMBINER_HPP
#define GNSS_COHERENT_COMBINER_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "gnssRecord.hpp"        // for RECORD_FLOATS + slot names (the record schema)
#include "prometheusMetrics.hpp" // for Counter, Gauge (real-time cost instrumentation)
#include "restServer.hpp"        // for connectionInstance

#include <array>   // for array (record export entries)
#include <complex> // for complex
#include <cstdint> // for int8_t
#include <mutex>   // for mutex
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class GnssCoherentCombiner
 * @brief The reassembly seam of the distributed-band pipeline.
 *
 * Each tracker emits, per PRN per window, the un-normalized coherent correlation @f$ G_m @f$
 * and replica energy @f$ E_m @f$ over its channel slice. The despread is a sum over channels,
 * so both are additive across the channel partition, and this stage forms the full-band
 * matched-filter amplitude
 *   @f$ \hat A = (\sum_m G_m) / (\sum_m E_m) @f$
 * which is identical to despreading all covering channels in one place.
 *
 * Inputs are aligned by each record's capture UTC (an input lagging the newest by more than
 * half a record is advanced). Records are summed by index, so every input must carry the same
 * PRN order. The output uses the combiner record layout of gnssRecord.hpp.
 *
 * Temporal integration over @c integration_length records gives both the incoherent amplitude
 * @f$ \sqrt{\langle|A|^2\rangle} @f$ (slot 3, ~sqrt(K) SNR, no phase needed) and the coherent
 * mean @f$ \langle A\rangle @f$ (slots 4-6, up to K SNR, but only while the carrier phase is
 * stable across the K records). Two cadences (@c integration_mode):
 *  - @c block (default): accumulate K records, emit once, reset.
 *  - @c rolling: an exponential moving average with time constant K records (alpha = 1/K),
 *    updated every record and emitted every @c output_every records without reset. Divided by
 *    1-(1-alpha)^n at emit, so it reads a true running mean from the first record.
 *
 * Slot 8 is a deep coherent |A| over the buffered per-record A, by one of four paths:
 *  - nav-bit wipe (@c navwipe_bit_records): records are grouped into data-bit epochs, each
 *    bit's +-1 is estimated and wiped, and the bits are summed coherently, so |A| keeps
 *    growing past the data bit where slot 6 decorrelates.
 *  - overlay wipe (@c secondary_overlay): a known secondary overlay is wiped at its
 *    best-fitting alignment.
 *  - both: the overlay wipe, then the nav-bit wipe on the result (a data component under a
 *    shared overlay, e.g. GPS L5-I).
 *  - plain coherent sum (@c deep_coherent), for a replica that already carries the overlay.
 * Each path is evaluated over a ladder of trailing windows (@c auto_coherence); a rung counts
 * as a coherent detection only if its SNR clears twice its noise-rectification floor (or, where
 * a rate search was applied, the search's peak/median gate).
 *
 * @conf n_prn               Int. Records (PRNs) per frame; default from the input frame size.
 * @conf n_elements          Int (default 0). Per-antenna blocks per PRN record; 0 = none.
 * @conf integration_length  Int (default 1). block: records per output. rolling: EMA time
 *                           constant in records, and the length of the deep window.
 * @conf integration_mode    String (default "block"). "block" or "rolling".
 * @conf output_every        Int (rolling only; default max(1, integration_length/10)). Records
 *                           between emits.
 * @conf navwipe_bit_records Int (default 0 = off). Records per data bit, e.g. 20 for 1 ms GPS
 *                           L1 C/A records. Needs integration_length >> this.
 * @conf secondary_overlay   String (default ""). Overlay name from the gnssOverlay registry.
 * @conf deep_coherent       Bool (default false). Deep plain coherent sum, used when no wipe is
 *                           configured; correct only when the despread replica already carries
 *                           the overlay.
 * @conf auto_coherence      Bool (default true). Also score octave-shorter trailing windows.
 * @conf carrier_pilot       Bool (default false). Dataless pilot: use the unsquared carrier
 *                           phase. Ignored when an overlay is configured (carrier_raw()).
 * @conf fft_len             Int (default 0). F-engine FFT length, to turn sample_seq into hops.
 * @conf deep_rate_search    Bool (default false). Phase-rate search before the plain sum.
 * @conf deep_rate_max_hz    Float (default 25). Cap on the applied rate; <= 0 = no cap.
 * @conf deep_rate_min_q     Float (default 10). Peak/median the rate search must reach.
 * @conf phase_track         Bool (default false). Leave-one-out phase-tracker candidates for
 *                           the plain sum.
 * @conf phase_track_widths  Int list (default [1, 2, 4, 8]). Tracker half-widths, records.
 * @conf sky_deep            Bool (default false). Sky-corrected prompt as a plain-sum candidate.
 * @conf record_export       Int (default 0 = off). Per-PRN record ring length for get_records.

 * @conf bit_export          Bool (default false). Publish nav_obs and bit_pred in get_status.
 * @conf bit_pred_horizon_s  Float (default 4). Seconds of predicted overlay chips.
 * @conf adr_smooth_records  Int (default 1). Records per ADR block for anchored overlay pilots.
 *
 * @buffer in_bufs Per-subband tracker records (gnssRecord.hpp tracker layout: a RECORD_FLOATS
 *                 header plus n_elements element blocks per PRN).
 * @buffer out_buf Combined records (gnssRecord.hpp combiner layout: 3 = |A|_incoh,
 *                 4,5 = <A>, 6 = |<A>|_coh, 7 = n_chan, 8 = deep |A|, 9-10 = UTC).
 *
 * @author Keith Vanderlinde
 */
class GnssCoherentCombiner : public kotekan::Stage {
public:
    GnssCoherentCombiner(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    void main_thread() override;

    static constexpr int RECORD_UTC_SLOT = gnss::RECORD_UTC_SLOT;

private:
    /// REST get_status: the last emit's per-PRN snapshot (amplitudes, deep detection, DLL
    /// powers, carrier and ADR observables).
    void get_status_callback(kotekan::connectionInstance& conn);
    /// REST get_records: per-record complex prompts for the fleet's coherent combine (see
    /// _rec_export); registered when record_export > 0.
    void get_records_callback(kotekan::connectionInstance& conn);
    /// REST get_elements: per-element complex-gain snapshot; registered when n_elements > 0.
    void get_elements_callback(kotekan::connectionInstance& conn);

    /// Per-bit observations from one navwipe evaluation, for the broker's navigation-message
    /// decoders. The bit at index @c bi covers record indices [bi*br - phase, (bi+1)*br - phase)
    /// counted from @c utc_ref, so its start UTC is utc_ref + (bi*br - phase)*rec_dt. Gaps in
    /// the record stream skip indices. The signs are the +-1 the deep wipe applies; their
    /// global polarity is arbitrary per emit.
    struct NavObs {
        double utc_ref = 0.0; ///< capture UTC of the window's first record
        double rec_dt = 0.0;  ///< median record spacing (s)
        int phase = 0;        ///< bit-epoch phase used (0..br-1)
        int br = 0;           ///< records per bit
        /// rot = e^{-i*theta0}, the phasor that aligned the bit sums with the real axis. The
        /// per-emit sign is arbitrary, but theta0 follows the carrier, so a consumer can chain
        /// polarity across emits by its continuity (navbit_predictor.py does).
        double rot_re = 1.0;
        double rot_im = 0.0;
        std::vector<std::pair<long long, int8_t>> bits; ///< (bit index, +-1 sign)
    };

    /// Nav-bit-wiped deep coherent amplitude from a window of per-record A and capture UTC.
    /// Records are binned into bit epochs by a record index taken from UTC relative to utc[0],
    /// so a dropped record leaves a gap rather than shifting later records. The epoch phase
    /// maximises the mean per-bit coherent magnitude; each bit's +-1 is taken against the
    /// squared-sum carrier phase, and the wiped bits are summed coherently. Returns the
    /// coherent mean per record, or 0 for a window of fewer than two bits.
    /// @param snr_out (optional) coherent sum / its noise std, the noise taken from the
    ///        component orthogonal to the aligned signal. Choosing each bit's sign rectifies
    ///        noise, so pure noise reads ~sqrt(2*n_bits/pi), not ~1.
    /// @param head (optional) per-record prompt over the hops before the record's code-period
    ///        boundary. A symbol edge can fall mid-record, and a record straddling a sign
    ///        change sums to |2f-1| of its amplitude; with @p head, record r's head goes to
    ///        record index cpi[r] and its tail a[r] - head[r] to cpi[r]+1, so bit sums are
    ///        built from symbol-aligned pieces.
    /// @param obs (optional) receives the per-bit decisions.
    /// @param pin_phase >= 0 skips the epoch search (navwipe_bit_records passes over the
    ///        window) and uses this phase; @p out_phase (optional) reports the phase used. The
    ///        epoch is a property of the signal, so a caller can search its longest window and
    ///        pin the shorter ones. The phase is window-relative (bit index =
    ///        (round((utc[r]-utc[0])/rec_dt) + phase)/br), so moving it to another window needs
    ///        the record-index offset llround((sub_utc0 - full_utc0)/rec_dt), taken from UTC and
    ///        never from buffer positions.
    double navwipe_amplitude(const std::vector<std::complex<double>>& a,
                             const std::vector<double>& utc, double* snr_out = nullptr,
                             const std::vector<std::complex<double>>* head = nullptr,
                             NavObs* obs = nullptr, int pin_phase = -1,
                             int* out_phase = nullptr) const;

    /// Residual carrier frequency (Hz) from a window of per-record A: a two-stage phase-slope
    /// fit over the whole window against capture UTC, so gaps keep their true span. A is
    /// squared to cancel +-1 nav-bit and overlay flips unless the stream has none (carrier_raw()
    /// or @p prewiped). Returns 0, which the broker's carrier loop reads as "skip", for a window
    /// under 16 records or a degenerate fit.
    /// @param sigma_phi_out (optional) weighted RMS of the fit residuals, radians on the true
    ///        carrier. Always written when non-null; NaN when no fit was made, never 0, which
    ///        would read as a perfect lock.
    /// @param prewiped the records are already overlay-free (the _navwipe stream), so the raw
    ///        phase is fitted: half the phase noise and no |2f-1| straddle null.
    /// @param snr_q_out (optional) per-record SNR s^2/N with the noise taken from the quadrature
    ///        component in the carrier-removed frame, so amplitude modulation (BOC pilots) does
    ///        not count as noise. Valid only while the fit holds over the window. Produced by
    ///        linear (unsquared) fits only; NaN otherwise. Divide by the record period for C/N0.
    double carrier_resid_hz(const std::vector<std::complex<double>>& a,
                            const std::vector<double>& utc, double* sigma_phi_out = nullptr,
                            bool prewiped = false, double* snr_q_out = nullptr) const;

    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    int _n_prn;

    /// Per-antenna blocks per PRN record (config n_elements); 0 = none, and the per-antenna
    /// path is skipped.
    int _n_elements = 0;
    int _rec_stride = gnss::RECORD_FLOATS;  ///< floats per PRN record, = record_stride(_n_elements)
    std::vector<double> _ge_r, _ge_i;       ///< scratch: per-antenna prompt summed over subbands
    std::vector<double> _acc_epow_el;       ///< [n_prn][n_elem] <|A_e|^2>, the beam-map estimator
    std::vector<double> _acc_ear, _acc_eai; ///< [n_prn][n_elem] <A_e>
    /// [n_prn][n_elem] per-element complex gain: A_e * conj(ref_e) accumulated per record, with
    /// ref_e the unweighted sum of the other elements' raw prompts (leave-one-out). The sky
    /// phase, common to all elements and near-white in time (gnssElemCal.hpp), cancels in each
    /// product before averaging; <A_e> averages that phase and collapses, so it cannot serve as
    /// the gain. Leaving the element out keeps its own noise out of its gain, and the raw
    /// prompts keep the element cal's weights out. _acc_euq accumulates the normalizer |ref_e|^2.
    std::vector<double> _acc_eu_r, _acc_eu_i, _acc_euq;
    /// REST snapshot of the element-gain state (under _st_mtx): [n_prn][n_elem][4] =
    /// u_re, u_im, <|A_e|^2>, <|ref|^2>, normalized at emit like the element blocks.
    std::vector<float> _st_elem;
    double _st_elem_keff = 0.0; ///< effective records behind the snapshot (significance input)
    int _integration_length;    ///< block: records/output; rolling: EMA time constant (records)
    bool _rolling;              ///< rolling EMA integration vs block-and-reset
    int _emit_every;            ///< rolling: records between emits (output cadence)
    int _navwipe_bit_records;   ///< records per nav bit (0 = no wipe)
    /// Shared secondary overlay (a registry row that is not per-PRN); empty if unused.
    std::vector<int8_t> _secondary;
    /// Per-PRN overlays (a per-PRN registry row such as B1C or L1CO), index prn-1; empty if
    /// unused.
    std::vector<std::vector<int8_t>> _l1co;
    bool _wipe_buffer = false; ///< buffer per-record A for a deep path (nav-bit, overlay or plain)
    /// Deep-integrate by a plain coherent sum with no wipe (config deep_coherent). For a pilot
    /// whose replica already carries the secondary overlay (CHORD despreads GPS_L5_Q_NH, NH20
    /// tiled into 204600 chips) there is nothing to wipe, and the wipe paths cannot serve it:
    /// overlay_apply advances the overlay one chip per record, while a CHORD record spans
    /// ~10.49 primary periods.
    bool _deep_plain = false;
    /// Subband UTC alignment in the gather loop: the observed record period, input 0's previous
    /// UTC, and the number of realignments (non-zero means an input dropped a frame relative to
    /// the others).
    double _rec_dt_est = 0.0;
    double _prev_utc0 = 0.0;
    unsigned long _realigns = 0;
    /// F-engine FFT length (samples per hop), used only to turn GnssChanMetadata::sample_seq, an
    /// absolute sample index, into the absolute hop index the rest of the chain uses (the fleet
    /// DLL groups on it; docs/CHORD_GNSS_SHARED_DLL.md). 0 = unset: pow_hop then carries the raw
    /// sample index, and pow_fft_len (also published) says which it is.
    int _fft_len = 0;
    bool _carrier_pilot; ///< config carrier_pilot; applies only through carrier_raw()
    /// Raw (unsquared) carrier phase is valid only for a dataless pilot with no secondary
    /// overlay: overlay chips left in the pre-wipe records flip the phase by pi like nav bits,
    /// so overlay pilots are squared like data signals.
    bool carrier_raw() const {
        return _carrier_pilot && _secondary.empty() && _l1co.empty();
    }
    /// Overlay chip sequence for a PRN: the per-PRN table if populated, else the shared
    /// secondary; nullptr when this combiner wipes no known overlay. The ADR path uses it to
    /// de-rotate the overlay linearly instead of squaring it away, which would double the
    /// carrier-phase noise. Same dispatch as the emit-time wipe.
    const std::vector<int8_t>* overlay_for(int prn) const {
        if (!_l1co.empty())
            return (prn >= 1 && prn <= (int)_l1co.size()) ? &_l1co[(size_t)(prn - 1)] : nullptr;
        return _secondary.empty() ? nullptr : &_secondary;
    }
    bool _auto_coherence; ///< also score octave-shorter trailing windows; the best one wins
    std::vector<std::vector<std::complex<double>>> _navbuf; ///< per-PRN per-record A, deep window
    /// Per-PRN per-record sky-phase-corrected A (record slots 24/25, gnssElemCal.hpp's
    /// leave-one-element-out derotation), parallel to _navbuf. All zero when the producer does
    /// not supply it, which is how the deep_coherent path decides whether the sky-corrected
    /// candidate is available.
    std::vector<std::vector<std::complex<double>>> _navsky;
    std::vector<std::vector<double>> _navutc; ///< per-PRN per-record capture UTC
    /// Per-PRN per-record head-segment amplitude (prompt over the hops before the record's
    /// code-period boundary), normalized by the total prompt energy so head + tail = A. The
    /// overlay can flip sign at that boundary, so the segmented wipes apply adjacent chips to
    /// head and tail. Equal to A when the tracker does not segment. Parallel to _navbuf.
    std::vector<std::vector<std::complex<double>>> _navhead;
    /// Per-PRN per-record overlay-wiped amplitude v_cur (segmented de-rotation, overlay-free at
    /// full amplitude for any boundary fraction), or 0 for records with no live anchor.
    /// Parallel to _navbuf. Feeds the carrier-residual fit linearly (prewiped), avoiding the
    /// squared raw A, which overlay transitions null at boundary fraction 0.5.
    std::vector<std::vector<std::complex<double>>> _navwipe;

    /// Publish the nav-bit wipe's per-bit decisions ("nav_obs") and a pilot's predicted overlay
    /// chips ("bit_pred") in get_status (config bit_export).
    bool _bit_export = false;
    std::vector<NavObs> _st_nav_obs; ///< last emit's observations per PRN (under _st_mtx)

    /// Real-time cost. The derived quantity is the duty cycle, CPU-seconds per second of data:
    /// (d ingest_us + d emit_us) / (d records x record period). Below 1 the stage keeps up; at 1
    /// it is saturated, more buffering only postpones the drops, and the drop rate stops
    /// measuring cost. Times are microseconds; the per-emit gauges show spikes the counters
    /// average away.
    kotekan::prometheus::Counter* _m_ingest_us = nullptr;    ///< cumulative record-ingest time
    kotekan::prometheus::Counter* _m_emit_us = nullptr;      ///< cumulative emit time
    kotekan::prometheus::Counter* _m_records = nullptr;      ///< records consumed (the data clock)
    kotekan::prometheus::Counter* _m_emits = nullptr;        ///< emits completed
    kotekan::prometheus::Gauge* _m_last_emit_us = nullptr;   ///< last emit's time
    kotekan::prometheus::Gauge* _m_last_emit_prns = nullptr; ///< PRNs with buffered deep records

    /// Predicted overlay chips for a pilot, projected forward from the dead-reckon anchor
    /// (_dr_phase, _dr_utc, _dr_rec_dt, the same anchor the ADR wipe uses). The overlay is
    /// deterministic, so once the anchor is pinned every later chip is known. Published as
    /// get_status "bit_pred" when bit_export is set, and only while the anchor is under 60 s
    /// old and _bp_agree holds. Assumes one primary period per record.
    struct BitPred {
        double utc0 = 0.0;        ///< projected start UTC of the first record at or after _st_utc
        double bit_s = 0.0;       ///< chip duration = record period
        std::vector<int8_t> bits; ///< +-1 overlay chips
        /// The table is record-indexed: bits[j] is the head chip of the j-th record from utc0,
        /// and utc0 is a record start, not a chip edge as in a time-domain table. Published as
        /// "grid": "record" so a consumer indexes it by record rather than by time.
        bool record_grid = true;
    };
    std::vector<BitPred> _st_bit_pred;
    double _bit_pred_horizon_s = 4.0; ///< seconds of chips to publish


    // Snapshot of the last emit for REST (under _st_mtx).
    std::vector<int> _st_prn;
    std::vector<float> _st_amp, _st_coh, _st_deep, _st_deep_snr, _st_amp_snr, _st_dop, _st_cp;
    std::vector<int> _st_nh_phase; ///< secondary-overlay alignment found per PRN (-1 = n/a)
    /// Count of emits whose anchor projection disagreed with the searched overlay phase, each
    /// of which suppressed that emit's bit_pred. Cumulative per PRN; published as "bp_veto"
    /// beside a published bit_pred.
    std::vector<int> _bp_veto;
    /// This emit's anchor projection agrees with the searched phase (1 when not judged).
    std::vector<uint8_t> _bp_agree;
    std::vector<float> _st_dll_disc; ///< window-averaged DLL discriminator (broker closes the loop)
    /// Fleet DLL (docs/CHORD_GNSS_SHARED_DLL.md): the window-averaged Early/Prompt/Late powers
    /// behind _st_dll_disc, published raw because ratios do not sum: the fleet discriminator
    /// (sum E - sum L)/(sum E + sum L) is not a function of the per-instance ones. All three
    /// are |sum of subband correlations|^2 / energy^2, so q = 2P/(E+L) is comparable across
    /// instances.
    std::vector<float> _st_e_pow;
    std::vector<float> _st_p_pow;
    std::vector<float> _st_l_pow;
    std::vector<float> _st_nchan; ///< covering channels that contributed (weight; 0 = dead)
    /// Absolute hop index labelling the E/P/L window (rolling: its newest record; block: its
    /// first); -1 = no frame metadata. The fleet combine groups instances by it; see _fft_len.
    long long _st_pow_hop = -1;
    /// Phase-rate search before the deep plain sum (config deep_rate_search). A rate is applied
    /// only when the spectrum's peak/median reaches _deep_rate_min_q, and only from bins within
    /// +-_deep_rate_max_hz.
    bool _deep_rate = false;
    double _deep_rate_max_hz;
    double _deep_rate_min_q = 10.0;
    std::vector<float> _st_deep_rate, _st_deep_rate_q;
    /// The rate search's uncapped argmax (parabolically interpolated) and its q: the carrier
    /// loop's measurement. deep_rate_hz is restricted to bins within deep_rate_max_hz, so a
    /// residual beyond the cap reads as the best in-cap noise bin and must not feed a loop.
    std::vector<float> _st_deep_rate_full, _st_deep_rate_full_q;

    /// Common-phase tracker before the deep plain sum (config phase_track): each record is
    /// derotated by the leave-one-out phase of its neighbours (gnss::phase_track_loo), which
    /// removes slow per-satellite phase wander. Candidates at each half-width in _pt_widths
    /// compete with the straight sum under the same estimator (the best width depends on SNR),
    /// and the floor pays for the selection.
    bool _phase_track = false;
    /// Score the record's sky-corrected prompt (slots 24/25) as one more deep candidate on the
    /// deep_coherent path (config sky_deep).
    bool _sky_deep = false;
    std::vector<int> _pt_widths; ///< half-widths tried (records); config
    /// |sum|/sum|.| of the winning deep stream (deep_coherent path only): a coherence measure
    /// that, unlike deep_snr, does not scale with the record count.
    std::vector<float> _st_coh_frac;
    std::vector<float> _st_pt_hw; ///< winning tracker half-width (0 = straight sum, -1 = sky)
    /// Per-record export for cross-node coherent combining (config record_export: ring length
    /// per PRN; 0 = off). Each entry is [hop, re, im, energy] for one record: the absolute hop
    /// is the exact cross-node alignment key, re/im the energy-normalized complex prompt, and
    /// energy the ML combining weight. _recex is the live ring (main thread); _st_recex its
    /// snapshot for REST.
    int _rec_export = 0;
    std::vector<std::vector<std::array<double, 4>>> _recex, _st_recex;
    /// Append one record to the export ring, trimming to _rec_export. Independent of the
    /// deep-window buffers, which are trimmed by _integration_length and absent in configs that
    /// do not wipe.
    void push_recex(int p, long long hop, double re, double im, double energy);
    std::vector<float> _st_head_frac; ///< boundary fraction f = <head energy>/<prompt energy>
    std::vector<float> _st_s4;        ///< amplitude scintillation index, thermal floor removed
    std::vector<float> _st_s4_raw;    ///< ... before the debias (diagnostic)
    std::vector<float> _st_sigma_phi; ///< carrier-phase jitter about the slope fit (rad)
    std::vector<float> _st_snr_q; ///< modulation-immune per-record SNR (quadrature noise); NaN n/a
    std::vector<float> _st_car_resid; ///< full-band carrier residual, Hz (shared carrier loop)
    /// Coherence time: span (s) of the chosen deep window; 0 when no ladder rung cleared its
    /// floor.
    std::vector<float> _st_coh_s;
    /// Noise-rectification floor (a significance) of the reported rung; deep_snr near it means
    /// no coherent detection.
    std::vector<float> _st_deep_floor;


    // Accumulated carrier phase (ADR), built per record from increments of
    // Phi_rx = Phi_cmd - arg(A)/2pi (gnssRecord.hpp): Phi_cmd's increment arrives in slot 15
    // and arg(A)'s comes from a cross-record phase product, so neither needs unwrapping. An arc
    // breaks on an inactive record or a record spacing of 50 ms or more, because unobserved
    // whole cycles are unknowable (a shorter gap is spanned by the product); each arc starts at
    // zero with its own integer ambiguity.
    std::vector<double> _trim_cyc; ///< commanded-trim integral this arc (cycles, slot 19), on
                                   ///< _adr_cyc's arc so downstream can subtract it
    std::vector<double> _adr_cyc;  ///< accumulated carrier phase this arc (cycles; double, as
                                   ///< float32 would quantize ~1e6 cycles to ~0.06)
    /// Residual half of the ADR alone: the sum of the measured per-record increments (dres) on
    /// _adr_cyc's arc. Its change over a poll divided by the record count is a fine
    /// carrier-rate measurement free of the large commanded integral.
    std::vector<double> _res_cyc;
    std::vector<double> _st_res;  ///< REST snapshot of _res_cyc, at emit
    std::vector<double> _adr_t0;  ///< arc start capture-UTC (lock time = utc - t0)
    std::vector<int> _adr_arc;    ///< arc id: ++ on every continuity break (slip counter)
    std::vector<int> _adr_n;      ///< records accumulated in this arc
    std::vector<uint8_t> _adr_ok; ///< phase continuity currently held
    /// Capture UTC of the emit these snapshots belong to, the epoch every observable is tagged
    /// with. The wall clock at poll time is not it: pipeline latency and emit jitter would enter
    /// any phase difference as f*dt_jitter (~6 Hz on a 2 kHz Doppler at 0.1 s of jitter).
    std::vector<double> _st_utc;
    std::vector<double> _st_adr;  ///< REST snapshot of _adr_cyc, at emit
    std::vector<double> _st_trim; ///< REST snapshot of _trim_cyc, at emit
    std::vector<double> _st_adr_lock;
    std::vector<int> _st_adr_arc, _st_adr_n;

    std::vector<int> _dr_phase;  ///< dead-reckon anchor: overlay phase at _dr_utc (-1 = none)
    std::vector<double> _dr_utc; ///< dead-reckon anchor capture-UTC (the winning rung's start)
    std::vector<int> _dr_prn;    ///< PRN the anchor belongs to (slot reassignment invalidates)
    /// Mean record spacing (s) over the anchor emit's winning rung, used to project the overlay
    /// chip index of any later record (ADR wipe, bit_pred). 0 = no anchor.
    std::vector<double> _dr_rec_dt;
    // ADR overlay wipe: the previous record's de-rotated amplitude, so consecutive records form
    // a plain product. Without an anchor (arc start, re-seed) the squared product is used.
    std::vector<std::complex<double>> _adr_vprev;
    std::vector<uint8_t> _adr_vprev_ok; ///< _adr_vprev is valid and continuous with this record

    /// K-record block ADR (config adr_smooth_records = K > 1; overlay pilots with a live anchor
    /// only). A per-record product folds by +-0.5 cycle on low-|A| records and random-walks the
    /// ADR; here de-rotated records accumulate into K-record block vectors and the increment is
    /// the cross-block product, which needs ~sqrt(K) more phase noise to fold. The product is
    /// memoryless, with a residual-frequency margin of +-0.5 cycle per block. Anchor loss or a
    /// 50 ms gap breaks the arc.
    int _adr_smooth = 1; ///< records per ADR block (1 = per-record product)
    std::vector<std::complex<double>> _adr_blk_v;    ///< current block's de-rotated vector sum
    std::vector<double> _adr_blk_dcmd;               ///< commanded-phase sum since last block close
    std::vector<int> _adr_blk_n;                     ///< records in the current block
    std::vector<std::complex<double>> _adr_blk_prev; ///< previous closed block vector
    std::vector<uint8_t> _adr_blk_prev_ok;
    std::vector<double> _adr_blk_prev_utc;
    std::vector<double> _adr_blk_rate;
    std::vector<uint8_t> _adr_blk_rate_ok;
    std::vector<int> _st_deep_rec; ///< records in the chosen deep window (= full window unless the
                                   ///< auto-coherence ladder found a shorter, more coherent one)

    /// Guards the _st_* snapshots; REST reads them from its own thread.
    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
