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
 *    updated every record and emitted every integration_length/10 records without reset. Divided by
 *    1-(1-alpha)^n at emit, so it reads a true running mean from the first record.
 *
 * Slot 8 is a deep coherent |A| over the buffered per-record A, by one of three paths:
 *  - nav-bit wipe (@c navwipe_bit_records): records are grouped into data-bit epochs, each
 *    bit's +-1 is estimated and wiped, and the bits are summed coherently, so |A| keeps
 *    growing past the data bit where slot 6 decorrelates.
 *  - overlay and nav-bit wipe (@c secondary_overlay with @c navwipe_bit_records): a known
 *    shared overlay is wiped at its best-ranked alignment, then the nav-bit wipe runs on the
 *    result (a data component under an overlay, e.g. GPS L5-I).
 *  - plain coherent sum (@c deep_coherent), for a replica that already carries the overlay.
 * Each path is evaluated over a ladder of octave-shorter trailing windows; a rung counts
 * as a coherent detection only if its SNR clears twice its noise-rectification floor (or, where
 * a rate search was applied, the search's peak/median gate).
 *
 * @conf n_prn               Int. Records (PRNs) per frame; default from the input frame size.
 * @conf n_elements          Int (default 0). Per-antenna blocks per PRN record; 0 = none.
 * @conf integration_length  Int (default 1). block: records per output. rolling: EMA time
 *                           constant in records, and the length of the deep window.
 * @conf integration_mode    String (default "block"). "block" or "rolling".
 * @conf navwipe_bit_records Int (default 0 = off). Records per data bit, e.g. 20 for 1 ms GPS
 *                           L1 C/A records. Needs integration_length >> this.
 * @conf secondary_overlay   String (default ""). Shared overlay name from the gnssOverlay
 *                           registry, wiped ahead of the nav-bit wipe; needs
 *                           navwipe_bit_records.
 * @conf deep_coherent       Bool. Must be true: without a nav-bit wipe the deep |A| is the plain
 *                           coherent sum, correct only when the despread replica already carries
 *                           the overlay. Default true.
 * @conf deep_rate_search    Bool. Must be true: the plain sum runs after a phase-rate search.
 *                           Default true.
 * @conf sky_deep            Bool. Must be true: the sky-corrected prompt is a plain-sum
 *                           candidate. Default true.
 * @conf deep_rate_min_q     Float. Must be 10, the peak/median the rate search must reach.
 *                           Default 10.
 * @conf phase_track         Bool. Must be false. Default false.
 * @conf fft_len             Int (default 0). F-engine FFT length, to turn sample_seq into hops.
 * @conf record_export       Int (default 0 = off). Per-PRN record ring length for get_records.
 * @conf bit_export          Bool (default false). Publish nav_obs in get_status.
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
    /// squared to cancel +-1 nav-bit and overlay flips. Returns 0, which the broker's carrier
    /// loop reads as "skip", for a window under 16 records or a degenerate fit.
    /// @param sigma_phi_out (optional) weighted RMS of the fit residuals, radians on the true
    ///        carrier. Always written when non-null; NaN when no fit was made, never 0, which
    ///        would read as a perfect lock.
    double carrier_resid_hz(const std::vector<std::complex<double>>& a,
                            const std::vector<double>& utc, double* sigma_phi_out = nullptr) const;

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
    /// Shared secondary overlay wiped ahead of the nav-bit wipe; empty if unused.
    std::vector<int8_t> _secondary;
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
    std::vector<std::vector<std::complex<double>>> _navbuf; ///< per-PRN per-record A, deep window
    /// Per-PRN per-record sky-phase-corrected A (record slots 24/25, gnssElemCal.hpp's
    /// leave-one-element-out derotation), parallel to _navbuf. All zero when the producer does
    /// not supply it, which is how the plain-sum path decides whether the sky-corrected
    /// candidate is available.
    std::vector<std::vector<std::complex<double>>> _navsky;
    std::vector<std::vector<double>> _navutc; ///< per-PRN per-record capture UTC
    /// Per-PRN per-record head-segment amplitude (prompt over the hops before the record's
    /// code-period boundary), normalized by the total prompt energy so head + tail = A. The
    /// overlay can flip sign at that boundary, so the segmented wipes apply adjacent chips to
    /// head and tail. Equal to A when the tracker does not segment. Parallel to _navbuf.
    std::vector<std::vector<std::complex<double>>> _navhead;
    /// Publish the nav-bit wipe's per-bit decisions ("nav_obs") in get_status (config
    /// bit_export).
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

    // Snapshot of the last emit for REST (under _st_mtx).
    std::vector<int> _st_prn;
    std::vector<float> _st_amp, _st_coh, _st_deep, _st_deep_snr, _st_amp_snr, _st_dop, _st_cp;
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
    /// The phase rate removed before the deep plain sum and its peak/median. A rate is applied
    /// only when the spectrum's peak/median reaches DEEP_RATE_MIN_Q, and only from bins within
    /// +-DEEP_RATE_MAX_HZ.
    std::vector<float> _st_deep_rate, _st_deep_rate_q;
    /// The rate search's uncapped argmax (parabolically interpolated) and its q: the carrier
    /// loop's measurement. deep_rate_hz is restricted to bins within DEEP_RATE_MAX_HZ, so a
    /// residual beyond the cap reads as the best in-cap noise bin and must not feed a loop.
    std::vector<float> _st_deep_rate_full, _st_deep_rate_full_q;
    /// |sum|/sum|.| of the winning deep stream (plain-sum path only): a coherence measure that,
    /// unlike deep_snr, does not scale with the record count.
    std::vector<float> _st_coh_frac;
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
    std::vector<int> _st_deep_rec; ///< records in the chosen deep window (= full window unless the
                                   ///< auto-coherence ladder found a shorter, more coherent one)

    /// Guards the _st_* snapshots; REST reads them from its own thread.
    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
