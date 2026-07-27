/**
 * @file
 * @brief Coherently recombine per-subband GNSS despread products into the
 *        full-band complex amplitude.
 *  - GnssCoherentCombiner : public kotekan::Stage
 */

#ifndef GNSS_COHERENT_COMBINER_HPP
#define GNSS_COHERENT_COMBINER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "gnssRecord.hpp"      // for RECORD_FLOATS + slot names (the record schema)
#include "restServer.hpp"      // for connectionInstance

#include <complex> // for complex
#include <cstdint> // for int8_t
#include <cstdio>  // for FILE (phase-dump instrumentation)
#include <mutex>   // for mutex
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class GnssCoherentCombiner
 * @brief The reassembly seam of the distributed-band pipeline.
 *
 * Each @ref GnssChannelizedTracker emits, per PRN per window, the un-normalized
 * coherent correlation @f$ G_m @f$ and replica energy @f$ E_m @f$ over its channel
 * slice. Because the despread is a sum over channels, both are additive across the
 * channel partition, so this stage forms the full-band matched-filter amplitude
 *   @f$ \hat A = (\sum_m G_m) / (\sum_m E_m) @f$
 * -- identical to despreading all covering channels in one place. That recovers
 * the full despread sensitivity from the per-subband (per-node, on CHORD) split.
 *
 * The trackers run lockstep on the same channelized windows, so the i-th frame of
 * every input is the same window and its records are in the same PRN order. The
 * output uses the standard GNSS record layout (matches @ref GnssChannelizedCorrelator)
 * so existing readers consume it unchanged.
 *
 * Optional temporal integration (@c integration_length records): the per-record
 * full-band amplitude is accumulated K ways before emission, giving both the
 * robust **incoherent** amplitude @f$ \sqrt{\langle|A|^2\rangle} @f$ (slot 3,
 * ~sqrt(K) SNR, no phase needed) and the **coherent** mean @f$ \langle A\rangle @f$
 * (slots 4/5/6, up to K SNR and unbiased, but only valid once the Doppler seed is
 * fine enough that the carrier phase is stable across the K records -- with a
 * coarse Doppler grid the coherent mean decorrelates and only slot 3 is usable).
 *
 * Two integration cadences (@c integration_mode):
 *  - @c block (default): accumulate K records, emit once, reset. One output per K
 *    records; the historical behaviour, byte-identical when the key is absent.
 *  - @c rolling: an exponential moving average with time constant K records
 *    (alpha = 1/K), updated every record and emitted every @c output_every records
 *    WITHOUT reset. Bias-corrected (divide by 1-(1-alpha)^n) so it reads a true
 *    running mean from the first record. Incoherent integration has no nav-bit cap
 *    and only needs the tracker to hold the code/Doppler bin, so a long rolling K
 *    (e.g. minutes of records) lets a weak sat climb out continuously -- you watch
 *    slot 3 (and the nav-wiped slot 8) grow instead of waiting K records per sample.
 *    The coherent slots 4-6 carry the same nav-bit limitation as in block mode.
 *
 * Optional **nav-bit wipe** (@c navwipe_bit_records > 0): coherent integration past
 * the 20 ms GPS data bit. Each record (one code period) lies wholly within one data
 * bit, so @f$ A_{rec} = d \cdot (\text{clean despread}) @f$ -- a constant +-1 sign. Over
 * the @c integration_length window the per-record A is buffered, grouped into
 * @c navwipe_bit_records-record bit epochs (alignment found by maximising per-bit
 * coherent power), the +-1 bit estimated per epoch by squaring (the global sign cancels
 * in |.|), wiped, and coherently summed -- giving a deep |A| that keeps growing past
 * 20 ms (slot 8), where the plain coherent mean (slot 6) decorrelates at the nav bit.
 *
 * @conf n_prn  Int. Records (PRNs) per frame; default from in-frame size.
 * @conf integration_length Int (default 1). block: tracker records accumulated per output.
 *       rolling: EMA time constant in records (the effective integration depth).
 * @conf integration_mode String (default "block"). "block" or "rolling" (see above).
 * @conf output_every Int (rolling only; default max(1, integration_length/10)). Records
 *       between rolling emits -- decouples the EMA update rate (every record) from the
 *       output/record cadence.
 * @conf navwipe_bit_records Int (default 0=off). Records per nav bit (~20 ms / record);
 *       e.g. 20 at 5 MSPS / 1 ms records. Needs integration_length >> this. In rolling mode
 *       the wipe runs over a sliding window of the last integration_length records.
 *
 * @buffer in_bufs Per-subband tracker record streams (RECORD_FLOATS floats/PRN:
 *                 0=PRN 1=dop 2=cp 3=corr.re 4=corr.im 5=energy 6=n_chan 9,10=UTC).
 * @buffer out_buf Combined records (0=PRN 1=dop 2=cp 3=|A|_incoh 4=<A>.re 5=<A>.im
 *                 6=|<A>|_coh 7=n_chan 8=|A|_navwipe 9,10=UTC).
 *
 * @author Keith Vanderlinde
 */
class GnssCoherentCombiner : public kotekan::Stage {
public:
    GnssCoherentCombiner(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GnssCoherentCombiner() override;
    void main_thread() override;

    static constexpr int RECORD_FLOATS = gnss::RECORD_FLOATS;     // schema: gnssRecord.hpp
    static constexpr int RECORD_UTC_SLOT = gnss::RECORD_UTC_SLOT;

private:
    /// broker poll: latest full-band |A| (and seed) per PRN, for drop decisions.
    void get_status_callback(kotekan::connectionInstance& conn);

    /// Nav-bit-wiped deep coherent amplitude from a window of per-record (A, capture-UTC):
    /// bin records into nav-bit epochs by their ABSOLUTE code-period index (from UTC, so
    /// valve drops just leave gaps, not misalignment), bit-sync, per-epoch +-1 by squaring,
    /// wipe, coherent-sum / N. 0 if too short. Needs capture-time UTC (capture_utc0 > 0).
    /// @c snr_out (optional): the deep detection's significance = coherent sum / its noise std
    /// (estimated from the component orthogonal to the aligned signal). deep == this SNR times its
    /// own uncertainty, so SNR >> 1 is a real lock, ~1 is noise.
    ///
    /// SEGMENTED (@c head non-null): the nav/CNAV symbol boundary is code-period-aligned at the
    /// transmitter but the record windows are hop-aligned, so it lands MID-record; a record
    /// straddling a symbol TRANSITION cancels to |2f-1| -- for L2C CM (ONE 20 ms symbol per
    /// record, bit_records 1) that is the same "bistable" null the overlay bands had. head[r]
    /// (the prompt over the hops before the boundary) belongs to period cpi[r], the tail to
    /// period cpi[r]+1, so the bit epochs are assembled from symbol-ALIGNED pieces (tail of one
    /// record + head of the next) and nothing cancels. head == a reduces bit-exactly to the
    /// unsegmented behaviour.
    /// Per-bit observations from one navwipe evaluation, for the broker's LNAV
    /// decode-and-predict (docs/gnss_voltage_peel_live.md P7a). The bit at index @c bi covers
    /// code periods [bi*br - phase, (bi+1)*br - phase) relative to @c utc_ref, i.e. its start
    /// UTC is utc_ref + (bi*br - phase)*rec_dt. Gaps in the record stream simply skip indices
    /// -- the pairs are honest about which bits were actually observed. The sign is the SAME
    /// +-1 the deep wipe applies; its global polarity is arbitrary (the LNAV parity/framing
    /// decode is polarity-independent, and the peel locks its gain to whatever frame the
    /// prediction uses), so no polarity convention is promised here.
    struct NavObs {
        double utc_ref = 0.0; ///< capture UTC of the window's first record
        double rec_dt = 0.0;  ///< median record period (s) -- the code-period duration
        int phase = 0;        ///< winning bit-phase (0..br-1)
        int br = 0;           ///< records per bit
        double rot_re = 1.0;  ///< the emit's polarity-resolving phasor (rot = e^{-i*theta0}):
        double rot_im = 0.0;  ///< each emit's global +-1 is arbitrary INDEPENDENTLY (the
                              ///< squaring estimate), but theta0 tracks the carrier, which IS
                              ///< continuous across emits at a certified lock -- the broker
                              ///< chains chunk polarity by phasor continuity, then VERIFIES by
                              ///< LNAV parity (and repairs by boundary-flip search).
        std::vector<std::pair<long long, int8_t>> bits; ///< (bit index, +-1 sign)
    };

    double navwipe_amplitude(const std::vector<std::complex<double>>& a,
                             const std::vector<double>& utc, double* snr_out = nullptr,
                             const std::vector<std::complex<double>>* head = nullptr,
                             NavObs* obs = nullptr) const;

    /// Residual carrier frequency (Hz) from a window of per-record A, as a bit-robust phase-SLOPE
    /// fit over the whole window (long baseline -> low variance -- the clean measurement the shared
    /// carrier loop needs; the old consecutive-record product was short-baseline + doubly-noisy and
    /// made the loop noise-inject). Squares A to cancel the +-1 nav-bit pi flips (data signals);
    /// a dataless pilot (carrier_pilot) fits the raw phase. Uses capture-UTC as the time axis so
    /// valve-drop gaps don't bias the slope. Returns 0 if too short / degenerate.
    /// @param sigma_phi_out (optional) weighted RMS of THIS fit's residuals, radians on the
    /// true carrier -- the phase-jitter half of the multipath/scintillation pair.
    /// @param prewiped the records are ALREADY overlay-free (the caller passes _navwipe = the
    /// de-rotated v_cur stream), so fit the RAW phase (mult 1, no squaring) exactly as for a
    /// dataless pilot -- half the phase noise AND no |2f-1| straddle null at boundary_f=0.5.
    double carrier_resid_hz(const std::vector<std::complex<double>>& a,
                            const std::vector<double>& utc,
                            double* sigma_phi_out = nullptr,
                            bool prewiped = false) const;

    std::vector<Buffer*> in_bufs;
    Buffer* out_buf;
    int _n_prn;
    int _integration_length; ///< block: records/output; rolling: EMA time constant (records)
    bool _rolling;           ///< rolling EMA integration vs block-and-reset
    int _emit_every;         ///< rolling: records between emits (output cadence)
    int _navwipe_bit_records; ///< records per nav bit (0 = no wipe)
    std::vector<int8_t> _secondary; ///< known PRN-independent overlay (L5 NH10/NH20); empty if unused
    std::vector<std::vector<int8_t>> _l1co; ///< per-PRN L1C-P overlays (index prn-1, 1..32); empty if unused
    bool _wipe_buffer = false;      ///< buffer per-record A for a deep wipe (navwipe or overlay)
    bool _carrier_pilot;            ///< pilot: unsquared phase product (no bits; 2x range)
    /// Raw (unsquared) phase treatment is only valid for a TRULY dataless pilot. An overlay
    /// pilot (B1C L1CO, E1C CS25, E5a/B2a CS100, L5 NH) still carries +-1 secondary chips in
    /// the PRE-WIPE navbuf records, which scramble a raw-phase fit exactly like nav bits:
    /// stage-1 pair products pick up random signs (coarse f biased to ~0) and stage-2 phases
    /// jump by pi. Measured 2026-07-17: the combiner reported C22's residual as 0.00-0.02 Hz
    /// while the true record-stream residual was +0.93 Hz -- the carrier loop was OPEN below
    /// a few Hz for every overlay pilot, leaving each sat's residual parked at seed error and
    /// certification waiting on LO-drift luck (the "45-min B1C bootstrap" + cert flicker).
    /// Squaring cancels +-1 overlay chips exactly as it cancels nav bits; 2x phase noise is
    /// the same price every data signal already pays.
    bool carrier_raw() const {
        return _carrier_pilot && _secondary.empty() && _l1co.empty();
    }
    /// Overlay chip sequence for a PRN: the per-PRN table (B1C/E5a/B2a/L1CO) if populated,
    /// else the shared secondary (L5 NH); nullptr when this combiner wipes no known overlay
    /// (data signals / navwipe). The ADR path uses it to de-rotate the overlay LINEARLY (a
    /// plain cross-record product) instead of squaring it away -- squaring doubles the
    /// carrier-phase noise and, at the low per-record SNR of the 1 ms L5-band records, is the
    /// L5-leg TEC scatter (see docs/tec_error_budget). Same dispatch as the emit-time wipe.
    const std::vector<int8_t>* overlay_for(int prn) const {
        if (!_l1co.empty())
            return (prn >= 1 && prn <= (int)_l1co.size()) ? &_l1co[(size_t)(prn - 1)] : nullptr;
        return _secondary.empty() ? nullptr : &_secondary;
    }
    bool _auto_coherence;           ///< deep wipe over an octave ladder of trailing sub-windows,
                                    ///< keep the best -> integrate as deep as the clock coheres
    std::vector<std::vector<std::complex<double>>> _navbuf; ///< per-PRN per-record A over the window
    std::vector<std::vector<double>> _navutc;              ///< per-PRN per-record capture UTC
    /// Per-PRN per-record HEAD-segment amplitude (prompt over the hops before the record's
    /// code-period boundary, normalized by the TOTAL prompt energy so head + tail = A).
    /// Feeds the SEGMENTED overlay wipe: the overlay flips sign at that boundary, so head
    /// and tail must be wiped with adjacent chips or straddling records cancel (the
    /// 2026-07-15 "bistable"). Parallel to _navbuf record-for-record.
    std::vector<std::vector<std::complex<double>>> _navhead;
    /// Per-PRN per-record OVERLAY-WIPED amplitude v_cur (segmented de-rotation; overlay-free at
    /// full amplitude for any boundary_f), or 0 for records with no live anchor. Parallel to
    /// _navbuf. Feeds the carrier-residual fit LINEARLY (prewiped) so the shared carrier loop no
    /// longer squares the raw A -- which cancels at boundary_f=0.5 and drove the loop open there,
    /// scrambling the whole window's phase (2026-07-22 B1C bf=0.5 deep-null root cause).
    std::vector<std::vector<std::complex<double>>> _navwipe;
    /// PEEL DEPTH (docs/gnss_voltage_peel_live.md). When the tracker peels, records carry a
    /// residual prompt (slots 20-23) beside the full correlation. Buffer it exactly like _navbuf/
    /// _navhead and run the SAME deep estimator on it at emit -> CMB_PEEL_DEEP; the ratio
    /// 20*log10(CMB_DEEP/CMB_PEEL_DEEP) is the achieved depth, and it is honest because the peel's
    /// gain was feed-forward. Gated on _peel_depth (config): only the peeling chain pays the
    /// second deep integration. Parallel to _navbuf record-for-record.
    bool _peel_depth = false;
    /// P7a: export the navwipe's per-bit sign decisions via get_status ("nav_obs"), so the
    /// broker can LNAV-decode and predict future bits for the peel. Config `bit_export: true`.
    bool _bit_export = false;
    std::vector<NavObs> _st_nav_obs; ///< last emit's observations per PRN (under _st_mtx)

    /// P7b PILOT OVERLAY PREDICTION. A pilot's secondary overlay (E1C CS25 / B1C / L1C-O /
    /// L5 NH) is DETERMINISTIC -- no decode needed, unlike LNAV. Once the dead-reckon anchor
    /// is pinned (_dr_phase/_dr_utc/_dr_rec_dt: the same anchor the ADR wipe projects from),
    /// every future chip is knowable, so the combiner can emit PREDICTIONS directly in the
    /// tracker's nav_bits format and the broker just forwards them -- the peel's sign source
    /// with ZERO new tracker code and no LNAV round trip. One chip per primary period, so
    /// bit_s == the record period and the chip boundary IS the code-period boundary the peel
    /// splits head/tail at.
    struct BitPred {
        double utc0 = 0.0;         ///< RECORD-START UTC of bits[0]'s primary period
        double bit_s = 0.0;        ///< chip duration = record period
        std::vector<int8_t> bits;  ///< +-1 overlay chips
        /// TABLE SEMANTICS, DECLARED (2026-07-27). This table is RECORD-INDEXED: cell j is the
        /// head chip of record i0+j, and utc0 is that record's START. It is NOT a time-domain
        /// table ("the chip covering time t"), which is what the GPS LNAV tables are (their
        /// utc0 is a true bit edge). The consumer must not guess: cudaGnssTrack reads
        /// record_grid tables by record index and time-domain tables by time.
        ///
        /// WHY THIS FIELD EXISTS. The 2026-07-26 fix made the record-grid table LOOK
        /// time-domain by publishing utc0 shifted by -(1-bf)*dt, so the consumer's
        /// half-record probe would land on the right cell. Measured 2026-07-27: that put the
        /// consumer's index expression at floor(N + psi + bf - 0.5) with psi + bf = 1.000 +-
        /// 0.02 on every satellite -- i.e. floor(N + 0.5), sitting EXACTLY on the truncation
        /// boundary, on every record, forever. Two errors tuned to cancel, on a discontinuity
        /// where the cancellation must be exact; satellites slipped a whole record and their
        /// peel died at any SNR (C33/C24 at 3000 sigma, 0.0 dB). Declaring the convention
        /// removes both the shift and the probe instead of balancing them.
        bool record_grid = true;
    };
    std::vector<BitPred> _st_bit_pred;
    double _bit_pred_horizon_s = 4.0;
    std::vector<std::vector<std::complex<double>>> _navbuf_res;  ///< residual prompt A per record
    std::vector<std::vector<std::complex<double>>> _navhead_res; ///< residual head segment

    /// Per-record phase-dump instrumentation (phase_dump_prns / phase_dump_path): for the listed
    /// PRNs append one text line per despread record -- capture-UTC, PRN, Re/Im of the full-band A,
    /// E^2/L^2 correlator powers, the commanded phase increment, and the seed dop/cp. The offline
    /// view INSIDE a deep window that the window-integrated sigma_phi cannot give (discrete
    /// half-cycle slips vs continuous wander, and the +-0.25-chip code-offset signature in E-L).
    /// Empty list (the default) = disabled, zero cost.
    std::vector<bool> _phase_dump_prn; ///< indexed by PRN number; true = dump this PRN
    FILE* _phase_dump = nullptr;       ///< open dump file (nullptr = disabled)

    // Latest combined record snapshot for REST status (full-band |A| per PRN).
    std::vector<int> _st_prn;
    std::vector<float> _st_amp, _st_coh, _st_deep, _st_deep_snr, _st_amp_snr, _st_amp_dbi, _st_dop,
        _st_cp;
    std::vector<int> _st_nh_phase; ///< secondary-overlay alignment found per PRN (-1 = n/a)
    /// bit_pred publications suppressed because the anchor's projected phase disagreed with
    /// this emit's SEARCHED phase (the pilot-side agreement gate; see the bit_pred block).
    /// Cumulative per PRN; exported as "bp_veto" so the broker/viewer can see a source that
    /// is being continuously refused rather than mistaking silence for health.
    std::vector<int> _bp_veto;
    std::vector<uint8_t> _bp_agree; ///< per-PRN: this emit's anchor projection == searched phase
    std::vector<float> _st_dll_disc; ///< window-averaged DLL discriminator (broker closes the loop)
    std::vector<float> _st_head_frac; ///< boundary fraction f = <head energy>/<prompt energy>
    std::vector<float> _st_s4;       ///< amplitude scintillation index, thermal floor removed
    std::vector<float> _st_s4_raw;   ///< ... before the debias (diagnostic)
    std::vector<float> _st_sigma_phi;///< carrier-phase jitter about the slope fit (rad)
    std::vector<float> _st_car_resid; ///< full-band carrier residual, Hz (shared carrier loop)
    std::vector<float> _st_coh_s;  ///< measured coherence: time span of the chosen deep window (s)
                                   ///< -- 0 when NO ladder rung beat its rectification floor
    std::vector<float> _st_deep_floor; ///< the reported rung's noise-rectification floor (sigma):
                                       ///< deep_snr ~ this value means NO coherent detection
    std::vector<float> _st_deep_pow; ///< fixed-full-window noise-debiased coherent power (Hz):
                                     ///< the map's unbiased coherent observable (mean 0 on noise)
    std::vector<float> _st_peel_deep;  ///< deep |A| of the PEEL RESIDUAL (0 when not peeling):
                                       ///< peel_depth_db = 20*log10(deep_amplitude/peel_deep)
    std::vector<float> _st_peel_incoh; ///< incoherent |A| of the peel residual
    std::vector<float> _st_peel_deep_snr; ///< the residual deep's SIGNIFICANCE -- compare against
                                          ///< deep_floor (same units) for the at-floor test;
                                          ///< peel_deep vs deep_floor is a UNITS ERROR
    // ---- Accumulated carrier phase (ADR): the precise ranging / TEC observable.
    // Reconstructed per RECORD as Phi_cmd (tracker, record slot 15) + arg(A)/2pi, accumulated
    // in increments so both ambiguities cancel (see the block in main_thread). Continuous
    // across f_ref re-pins by construction; an arc BREAKS on any gap or inactive record,
    // because unobserved whole cycles are unknowable. Each arc carries its own integer
    // ambiguity, so each begins at zero and downstream levels it against the code phase.
    std::vector<double> _trim_cyc;     ///< commanded-trim integral this arc (cycles; slot 19 --
                                       ///< same arc lifecycle as _adr_cyc so downstream can
                                       ///< subtract exactly; docs/adr_trim_subtraction.md)
    std::vector<double> _adr_cyc;      ///< accumulated carrier phase this arc (cycles; DOUBLE --
                                       ///< float32 would quantize ~1e6 cycles to 0.06)
    std::vector<double> _adr_cph_prev; ///< previous record's commanded phase (cycles mod 1)
    std::vector<double> _adr_rate;     ///< commanded-phase rate (Hz) -- the unwrap predictor. The
                                       ///< reported Doppler errs by 2*trim, which at a 10 ms B1C
                                       ///< record is 2 whole cycles: it would unwrap to the wrong
                                       ///< integer and never come back.
    std::vector<double> _adr_t0;       ///< arc start capture-UTC (lock time = utc - t0)
    std::vector<int> _adr_arc;         ///< arc id: ++ on every continuity break (slip counter)
    std::vector<int> _adr_n;           ///< records accumulated in this arc
    std::vector<uint8_t> _adr_ok;      ///< phase continuity currently held
    std::vector<double> _st_utc;       ///< CAPTURE UTC of the emit these snapshots belong to --
                                       ///< the epoch every observable is tagged with. Wall-clock
                                       ///< at poll time is NOT it (pipeline latency + emit
                                       ///< cadence jitter): differencing phase against wall time
                                       ///< injects f*dt_jitter, ~6 Hz on a 2 kHz Doppler at 0.1 s
                                       ///< of jitter -- which is exactly what the first ADR
                                       ///< acceptance run measured before this field existed.
    std::vector<double> _st_adr;       ///< REST snapshots of the above, at emit
    std::vector<double> _st_trim;      ///< REST snapshot of _trim_cyc, at emit
    std::vector<double> _st_adr_lock;
    std::vector<int> _st_adr_arc, _st_adr_n;

    std::vector<int> _dr_phase;   ///< dead-reckon anchor: overlay phase at _dr_utc (-1 = none)
    std::vector<double> _dr_utc;  ///< dead-reckon anchor capture-UTC (the winning rung's start)
    std::vector<int> _dr_prn;     ///< PRN the anchor belongs to (slot reassignment invalidates)
    std::vector<double> _dr_rec_dt; ///< record period (s) measured at the anchor emit -- projects
                                    ///< the overlay chip index of any LATER record so the ADR path
                                    ///< can de-rotate the overlay per-record (accumulation time,
                                    ///< before the next emit's alignment search). 0 = no anchor yet.
    // ADR overlay-wipe path: the previous record's overlay-corrected (de-rotated) amplitude,
    // so consecutive records form a PLAIN product (no squaring). Reduces to the squared path
    // whenever no anchor is available (arc start before the first locked emit, or a re-seed).
    std::vector<std::complex<double>> _adr_vprev;
    std::vector<uint8_t> _adr_vprev_ok; ///< _adr_vprev is valid + continuous with this record

    /// K-RECORD SMOOTHED-VECTOR ADR (adr_smooth_records, 2026-07-22: the fold-walk fix).
    /// The per-1ms-record cross-record product folds (+-0.5-cycle ambiguity) during low-|A|
    /// records and random-walks the ADR: measured 26.5 m live vs 1.17 m in the same records'
    /// smoothed phase (G26, then C33 8h cross-constellation confirm; ratio 37-84x). With
    /// _adr_smooth = K > 1 (overlay pilots with a live anchor only), de-rotated records
    /// accumulate into K-record block vectors and the ADR increment comes from the
    /// cross-BLOCK product: noise folds need ~sqrt(K) more phase noise, while the residual-
    /// frequency fold margin (+-0.5 cyc per K records) stays >= 25 Hz at K=20 -- above the
    /// 17 Hz worst-case transient. Anchor loss / gaps break the arc honestly (no mode mixing).
    int _adr_smooth = 1; ///< records per ADR block (1 = legacy per-record product)
    std::vector<std::complex<double>> _adr_blk_v;   ///< current block's de-rotated vector sum
    std::vector<double> _adr_blk_dcmd;              ///< commanded-phase sum since last block close
    std::vector<int> _adr_blk_n;                    ///< records in the current block
    std::vector<std::complex<double>> _adr_blk_prev; ///< previous closed block vector
    std::vector<uint8_t> _adr_blk_prev_ok;
    std::vector<double> _adr_blk_prev_utc;
    /// Rate-predicted block unwrap (v2, 2026-07-22 morning): the first soak showed the raw
    /// block product's +-0.5-cycle-per-block margin (+-25 Hz at K=20) is EXCEEDED by the
    /// standing 13-25 Hz carrier residuals some L5-band sats carry (strong sats included --
    /// a loop-health question, not SNR). Unwrapping each block product around an EMA of the
    /// recent block rate tolerates any STEADY residual; only a rate CHANGE > half a cycle
    /// per block between consecutive blocks (~>2500 Hz/s at K=10) could still fold.
    std::vector<double> _adr_blk_rate; ///< EMA of the block-to-block residual step (cycles)
    std::vector<uint8_t> _adr_blk_rate_ok;
    std::vector<int> _st_deep_rec; ///< records in the chosen deep window (= full window unless the
                                   ///< auto-coherence ladder found a shorter, more coherent one)

    /// nh TIME-ASSIST (nh_assist: true). The broker (which has the almanac) POSTs, per PRN, the
    /// PREDICTED absolute overlay-chip index from BeiDou time + range -- an emit-independent
    /// number in the broker's own convention. Each emit we measure the convention/window offset
    /// from the sats the blind search LOCKED CONFIDENTLY (offset = nh_found - hint, sat-independent
    /// at one emit), then seed the deep wipe of the sats that DIDN'T clear at (hint + offset): the
    /// weak sats that cannot win the 1800-way alignment search get the geometrically-correct
    /// alignment for free. Purely additive: a wrong hint just fails its floor and the blind result
    /// stands. -1 = no hint yet.
    bool _nh_assist = false;
    int _nh_min_refs = 3;         ///< min confidently-locked sats that must AGREE on the offset
    int _nh_search_chips = 2;     ///< hinted-wipe half-width (chips) around hint+offset. +-2 is
                                  ///< right for the BRDC almanac (m-accurate range + sat clock =
                                  ///< sub-chip hints); the TLE era needed +-8 to survive the
                                  ///< celestrak IGSO label rot (5-7 chips). Narrower = lower
                                  ///< selection floor + cheaper.
    double _nh_ref_margin = 2.0;  ///< reference bar as multiple of the wipe floor (2.0 = the
                                  ///< certification gate itself; cluster agreement is the guard)
    std::vector<int> _nh_hint;    ///< broker's predicted absolute overlay index per PRN (-1 = none)
    std::mutex _nh_mtx;           ///< guards _nh_hint (POST callback vs main thread)
    void set_nh_hint_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
