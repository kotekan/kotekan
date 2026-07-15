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
    double navwipe_amplitude(const std::vector<std::complex<double>>& a,
                             const std::vector<double>& utc, double* snr_out = nullptr,
                             const std::vector<std::complex<double>>* head = nullptr) const;

    /// Residual carrier frequency (Hz) from a window of per-record A, as a bit-robust phase-SLOPE
    /// fit over the whole window (long baseline -> low variance -- the clean measurement the shared
    /// carrier loop needs; the old consecutive-record product was short-baseline + doubly-noisy and
    /// made the loop noise-inject). Squares A to cancel the +-1 nav-bit pi flips (data signals);
    /// a dataless pilot (carrier_pilot) fits the raw phase. Uses capture-UTC as the time axis so
    /// valve-drop gaps don't bias the slope. Returns 0 if too short / degenerate.
    /// @param sigma_phi_out (optional) weighted RMS of THIS fit's residuals, radians on the
    /// true carrier -- the phase-jitter half of the multipath/scintillation pair.
    double carrier_resid_hz(const std::vector<std::complex<double>>& a,
                            const std::vector<double>& utc,
                            double* sigma_phi_out = nullptr) const;

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
    // ---- Accumulated carrier phase (ADR): the precise ranging / TEC observable.
    // Reconstructed per RECORD as Phi_cmd (tracker, record slot 15) + arg(A)/2pi, accumulated
    // in increments so both ambiguities cancel (see the block in main_thread). Continuous
    // across f_ref re-pins by construction; an arc BREAKS on any gap or inactive record,
    // because unobserved whole cycles are unknowable. Each arc carries its own integer
    // ambiguity, so each begins at zero and downstream levels it against the code phase.
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
    std::vector<double> _st_adr_lock;
    std::vector<int> _st_adr_arc, _st_adr_n;

    std::vector<int> _dr_phase;   ///< dead-reckon anchor: overlay phase at _dr_utc (-1 = none)
    std::vector<double> _dr_utc;  ///< dead-reckon anchor capture-UTC (the winning rung's start)
    std::vector<int> _dr_prn;     ///< PRN the anchor belongs to (slot reassignment invalidates)
    std::vector<int> _st_deep_rec; ///< records in the chosen deep window (= full window unless the
                                   ///< auto-coherence ladder found a shorter, more coherent one)
    std::mutex _st_mtx;
};

#endif // GNSS_COHERENT_COMBINER_HPP
