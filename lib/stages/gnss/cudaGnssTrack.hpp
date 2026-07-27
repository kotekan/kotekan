#ifndef CUDA_GNSS_TRACK_HPP
#define CUDA_GNSS_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "gnssChannelizedReplica.hpp"
#include "restServer.hpp"

#include <atomic>
#include <complex>
#include <fstream>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

/**
 * Shared state for @ref cudaGnssTrack (one per stage, shared by the per-frame command
 * instances): the replica bank + GPU despread driver, the broker-facing seed set (REST
 * /set_seeds, same contract as GnssChannelizedTracker), the tracking control state (pinned
 * f_ref per PRN, fence anchors), and the device ring bookkeeping. All of pass-1's decisions
 * live HERE; the per-instance command objects are stateless dispatchers.
 */
class cudaGnssTrackState : public cudaCommandState {
public:
    cudaGnssTrackState(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssTrackState();

    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    // Geometry / config (immutable after construction).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, chan_offset = 0, fft_len = 0, hops_per_record = 0;
    double sample_rate = 5e6, capture_utc0 = 0.0, doppler_margin_hz = 5000.0;
    double dll_spacing = 0.5, fll_reacq_hz = 200.0, max_anchor_age_s = 30.0;
    double f_offset_hz = 0.0; ///< carrier IF (bin-distance reference for max_cover_bins)
    int max_cover_bins = 0;   ///< DIAGNOSTIC (2026-07-21, L5 ADR-wander channel-width A/B):
                              ///< despread on only the N covering channels nearest the
                              ///< carrier. 0 (default) = the full covering set.
    long long ring_hops = 0; ///< multiple of hops_per_record (windows never straddle the wrap)
    bool quantized = false;  ///< input frames + device ring are CHORD 4+4b bytes (GnssQuantize44
                             ///< upstream); scales arrive via GnssChanMetadata::chan_scale

    // 4+4b dequantization scales (lsb -> volts): the last set seen on frame metadata, staged
    // here for the async upload to the device array (pageable staging = host-synchronous on
    // return, same pattern as the jobs). Empty until the quantizer freezes its bandpass.
    std::vector<float> chan_scale_host;
    bool chan_scale_valid = false; ///< device array holds chan_scale_host

    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;

    // Broker seeds (REST-updated; snapshotted per GPU frame under the mutex).
    std::mutex seed_mtx;
    std::vector<double> dop, cp, cp_rate, dop_rate, ctrim;
    std::vector<long long> ref_hop;
    std::vector<uint8_t> active;

    // Tracking control state (pass-1 owns it; command instances execute in frame order).
    std::vector<double> f_ref;         ///< pinned replica carrier per PRN (NaN = unset)
    std::vector<long long> reacq_hop;  ///< ff_hz ramp anchor per PRN

    // ---- VOLTAGE PEEL (docs/gnss_voltage_peel_live.md) ----
    bool peel = false;         ///< subtract each seeded PRN's waveform before the despread
    double peel_alpha = 0.01;  ///< LMS/EMA rate on the DEROTATED gain (~1/alpha records)
    double peel_min_amp = 0.0; ///< |a| gate: below this the PRN is not peeled at all
    /// GAIN-QUALITY GATE (amplitude SNR of the AVERAGED gain). The peel removes a*R and
    /// leaves (a - a_hat)*R, so the residual is |e|/|a| = 1/SNR of the gain ESTIMATE: at
    /// SNR 1 the peel does nothing, and BELOW 1 it ADDS power. Measured live 2026-07-24 with
    /// no gate (peel_min_amp 0): 21 of 44 sats sat at |resid|/|full| ~1.57, i.e. the peel was
    /// making the band DIRTIER on half the constellation -- the exact opposite of the point.
    /// Peel only where the gain is good enough to help by a real margin.
    double peel_min_gain_snr = 3.0;
    /// SIGN-QUALITY GATE -- the gate above CANNOT see this failure. Measured live 2026-07-24:
    /// on GPS L1CA the gain-SNR gate rejected NOBODY, yet 6 of 12 sats sat at |resid|/|full|
    /// 1.3-2.2 (adding power), split perfectly by whether the broker had a nav-bit prediction:
    ///   predicted {3,16,26,28,29,31,32} -> 0.15-0.4     no prediction {4,14,15,20,22,25} -> >1.3
    /// The reason it is invisible: the gain is measured DE-BITTED by each record's OWN observed
    /// sign, so it is clean and high-SNR either way; the damage is done at SUBTRACTION time by a
    /// sign that is one GPU frame (~10 records) stale and therefore uncorrelated with the 20 ms
    /// bit. A variance gate sees random error, not systematic sign error. The old claim that
    /// "same as last" is right ~19 records in 20 assumed a ONE-RECORD lag; at a one-FRAME lag it
    /// is worse than a coin flip (ratio >1 means anti-correlated). So: no prediction, no
    /// subtraction. Pilots are unaffected -- their overlay is deterministic and always predicted.
    bool peel_require_bits = true;
    std::vector<double> gain_p2; ///< [n_prn] EMA of the per-record gain POWER (signal+noise);
                                 ///< with |EMA gain|^2 it separates the two -> the SNR above
    /// ...but that pair is PER-CHANNEL, and both the subtraction residual and peel_ratio are
    /// COHERENT SUMS over channels, which buy ~sqrt(n_chan) against (independent) channel noise.
    /// Estimating the SNR per channel therefore understates it by that factor -- it made the
    /// gate ~3x too strict at n_chan=10 and made the reported floor read ABOVE ratios it was
    /// supposed to bound (impossible; that is how the bug surfaced, 2026-07-24). So track the
    /// SUMMED correlation C = sum_c a_c*E_c in its own EMA, already in the space the ratio
    /// lives in. <|C|^2> is signal power PLUS per-record noise power; the noise term below
    /// separates them (signal = <|C|^2> - sigma^2).
    std::vector<double> gain_sum_p2;             ///< [n_prn] EMA of |C|^2 (signal + noise)
    /// Per-record noise from SUCCESSIVE DIFFERENCES: sigma^2 = <|C_k - C_{k-1}|^2>/2.
    /// The obvious estimator -- <|C|^2> - |<C>|^2 -- uses a COHERENT mean, so any slow wander
    /// of the summed gain phasor shrinks |<C>| below |C| and is then counted TWICE: once
    /// deflating the signal term, once inflating the noise term. Measured 2026-07-25 over an
    /// 8h45m soak, that pushed the floor above the residual it is supposed to bound --
    /// `peel_ratio/floor` reached 0.35 and sat at a median of 0.90 on GPS, where a
    /// thermal-limited residual cannot go below 0.886 (Rayleigh mean of complex noise).
    /// A difference cancels anything slowly-varying, signal phase included, so this term is
    /// immune to the drift; it DOES retain fast record-to-record rotation, which genuinely
    /// limits what the subtraction can achieve and belongs in the floor.
    std::vector<std::complex<double>> gain_sum_prev; ///< [n_prn] previous record's C
    std::vector<double> gain_dvar;                   ///< [n_prn] EMA of |dC|^2/2 = sigma^2
    int peel_warmup = 100;     ///< gain updates before a PRN's gain is trusted enough to subtract

    /// Feed-forward gain per (PRN, channel), in the NCO-DEROTATED frame -- which is where it is
    /// SLOWLY VARYING, and therefore where averaging is legitimate. Per channel because it costs
    /// nothing (the despread already reports per channel) and it absorbs the receiver bandpass,
    /// which is not in the replica and which a single combined gain cannot represent.
    std::vector<std::complex<float>> gain; ///< [n_prn][n_chan]
    std::vector<int> gain_n;               ///< updates so far, per PRN (vs peel_warmup)
    /// P7a PREDICTED NAV BITS (broker nav_bits, from the LNAV decode-and-predict): the +-1
    /// sign per 20 ms bit on the capture clock, utc0-anchored. 0 = unknown -> the
    /// decision-directed fallback below. Replaces the one-frame-stale gain_sign wherever
    /// known -- the removal of the ~11 dB sign-lag depth ceiling on data signals.
    struct NavBits {
        double utc0 = 0.0, bit_s = 0.02;
        std::vector<int8_t> bits;
    };
    std::vector<NavBits> nav_bits; ///< [n_prn], seed-updated under seed_mtx
    /// Which COMPONENT of the seed row's nav_bits this chain consumes. The wire schema is
    /// component-keyed -- nav_bits: {"P": {...}, "D": {...}} -- where the keys are RELATIONAL:
    /// "P" is the component this chain's replica correlates (the pilot on pilot chains, the
    /// sole component on L1CA), "D" the paired data sibling (E1B beside E1C, B1Cd beside
    /// B1Cp, L1Cd beside L1Cp, ...). A bare single table is accepted as "P" for back-compat.
    /// Relational rather than signal-named because the broker does not universally know the
    /// tracker's signal string, and roles survive across constellations. When the data-channel
    /// peel lands, the second component becomes a second PeelSpec row with its own table --
    /// the schema is fixed NOW, while there is exactly one producer and one consumer, so that
    /// change is additive instead of breaking.
    std::string peel_component = "P";
    /// DIAGNOSTIC: despread the RAW window beside the peeled+added-back one and log the
    /// difference. Answers "is the analytic add-back exact on this chain" with a number
    /// instead of an argument. Costs an extra despread + a blocking readback -- never on in
    /// production. See the allocation-site note in the .cpp.
    bool peel_verify = false;
    /// Verify one record in N (the readback blocks the stream; every record would stall
    /// every chain sharing this GPU process).
    int peel_verify_every = 500;
    /// Rows whose nav_bits failed to parse, since the last health line. This used to be
    /// swallowed in silence ("keep the previous table"), and on 2026-07-25 that cost a live
    /// debugging session: the constructed-bit source took every GPS PRN to `nobits` at once --
    /// including the decoded ones that never used it -- while the broker reported a clean
    /// `seeded 1/1 trackers` and the POST returned 200. A silent catch on a path that feeds
    /// SIGNS to a subtracter is not harmless; it is exactly where a fault hides.
    std::atomic<int> nav_bits_bad{0};

    /// Last MEASURED overlay/nav sign, per PRN, and the prediction used for the next record.
    /// The EMA above is DE-BITTED (the sign is divided out before averaging), so the sign has to
    /// be carried separately and re-applied at subtraction time. Predicting "same as last" is
    /// exact for a pilot and right ~19 records in 20 on GPS nav data.
    std::vector<double> gain_sign;         ///< [n_prn], +-1

    /// NCO phase MIRROR. The gain is averaged derotated but must be subtracted in the replica
    /// frame, so the peel needs this record's NCO phase -- the same integration
    /// GnssGpuRecordAssemble runs (re-pin fold included). Deliberately duplicated rather than
    /// plumbed across stages: every input (f_nco, fcar, reanchored, wstart) is computed HERE in
    /// pass 1, and a REST/shared-state hop would add a frame of latency to a per-record quantity.
    /// ⚠️ If the assembler's NCO integration changes, change this with it.
    std::vector<double> phi_nco;     ///< [n_prn] radians, wrapped
    std::vector<double> fcar_prev;   ///< [n_prn] previous record's replica f_ref (re-pin fold)
    std::vector<uint8_t> nco_ok;     ///< [n_prn] phase history valid
    std::vector<long long> wstart_prev_p; ///< [n_prn] previous record's window start (sample)

    /// PEEL FLL: the NCO mirror above carries only the COMMANDED carrier (ctrim + the ramp FF).
    /// The residual (true - commanded) is ~0.01 Hz once the broker's shared carrier loop is
    /// closed, but Hz-level before it converges -- and at Hz-level the gain EMA averages a
    /// ROTATING phasor: the magnitude survives at reduced scale while the phase lags ~90 deg, so
    /// the subtraction removes the right amount at the wrong angle, i.e. ~nothing (measured
    /// 2026-07-24 on the broker-less bench: |g| converged and stable, depth pinned at 0 dB).
    /// Track the residual with a per-PRN FLL on the record-to-record phase of the gain
    /// measurements (bit-robust squared discriminator, the codebase's standard), integrate it
    /// into the derotation, and the EMA sees a stationary phasor again. On the live node with
    /// ctrim closed this loop idles near zero; on a bench (or during broker convergence) it is
    /// what makes the peel work at all. Same design as the CPU peel v2's f_track/phi_track.
    double peel_fll_gain = 0.05;
    /// Alias-capture guards for the loop above. The squared discriminator determines the
    /// residual only modulo 1/(2*dt_rec), so without a bound the integrator walks the alias
    /// ladder (measured to 19 kHz overnight). 10 Hz sits far inside the first rung on every
    /// live chain (BDS/L1C 50, GAL 125, GPS 500 Hz) and far above the real residual
    /// (~0.01 Hz with ctrim closed, Hz-level while the broker converges).
    double peel_fll_max_hz = 10.0;
    double peel_fll_max_slew_hz = 2.0; ///< per-update step limit (bench cure, e20bc755)
    std::string peel_tag;      ///< chain identity for the health line (multi-chain node)
    /// GROUND TRUTH per PRN: EMA of |residual|/|full| measured on the SAME record, straight
    /// from the mirrored despread rows. 0 = perfect cancellation, 1 = nothing subtracted.
    /// Independent of the combiner's deep estimators, so it separates "the kernel is not
    /// subtracting" from "the depth measurement is wrong" -- the exact ambiguity that cost
    /// two wrong theories on 2026-07-24.
    std::vector<double> peel_ratio;
    std::vector<int> peel_ratio_n;
    /// ...and the SAME ratio built from per-channel MAGNITUDES, sqrt(sum|res_c|^2/sum|full_c|^2),
    /// instead of the coherent channel sum |sum res_c|/|sum full_c| above.
    ///
    /// WHY BOTH (2026-07-25). The coherent version divides by a sum of complex per-channel
    /// correlations, so on a PHASE-DIVERSE satellite the DENOMINATOR self-cancels and the ratio
    /// is driven toward 1 -- which is indistinguishable from "the peel subtracted nothing", the
    /// exact signature that had 7 of 8 GAL sats reading a flat 1.00 while their tracking was
    /// pristine. Summing magnitudes cannot self-cancel, so this one measures the peel even when
    /// the other is measuring the channel geometry. The pair is the instrument:
    ///   both ~1        -> the peel really is not cancelling (look at the gain / the phase)
    ///   ic << 1, coh~1 -> the peel WORKS and the coherent metric (and the floor, which shares
    ///                     the defect -- open item 0a, GAL PRN2 floor 74 on 07-24) was lying.
    /// Costs nothing new: both numbers come from mirror rows already resident on the host.
    std::vector<double> peel_ratio_ic;
    /// PER-RECORD PHASE RESIDUAL after derotation: d = arg(sum_c a_c * conj(g_c)), i.e. how far
    /// this record's measured phasor sits from the gain the EMA has already built. Two EMAs:
    /// pres2 = <d^2> (the LEVEL) and pres_d2 = <(d - d_prev)^2> (the record-to-record
    /// INCREMENT). Reported in the health line as p<rms_deg>d<incr_rms_deg>.
    ///
    /// WHY (2026-07-25). deep_snr PREDICTS what the gain coherence must be: the combiner sums
    /// the same per-record phasors coherently and reaches 186 sigma over 250 records, so
    /// gcoh should be ~1/(1 + N/deep_snr^2) = 0.99. Measured 0.00. The signal is therefore
    /// provably coherent and the MIRROR's phase is wrong by order a radian per record. These
    /// two numbers say which kind of wrong, without another guess:
    ///   p large, d large   -> the phase scrambles RECORD TO RECORD (bookkeeping: the mirror
    ///                         disagrees with the assembler per record)
    ///   p large, d small   -> a slow drift / lag: an uncompensated FREQUENCY the FLL is not
    ///                         absorbing (the EMA trails a rotating phasor)
    ///   both small         -> the phase is fine and gcoh is measuring something else, which
    ///                         would mean my own gcoh is the faulty instrument
    /// Fully random phase gives p = 180/sqrt(3) = 104 deg, d = 147 deg -- the scale to read
    /// these against.
    std::vector<double> pres2, pres_d2, pres_prev;
    std::vector<uint8_t> pres_ok;
    /// ⚠️ pres2 ALONE CANNOT NAME THE FAULT. A wrong bit/overlay sign is a phase error of
    /// exactly pi, so it lands in the same RMS bucket as genuine phase scatter: uniform random
    /// phase gives RMS 104 deg, a 50/50 sign flip gives 127 deg, a ~35% flip gives ~105 deg --
    /// and the flat population measured 105. Three different faults, one number. These two
    /// separate them, because cos(2d) is BLIND to a pi flip and sensitive to everything else:
    ///     <cos d>   <cos 2d>
    ///        0          0      -> uniform random PHASE (the derotation is wrong)
    ///      1-2q         1      -> SIGN flips on a fraction q of records (the bits are wrong),
    ///                            phase otherwise clean
    ///     small       middling -> both, or a phase error that is neither
    /// Sign was "cleared" earlier by checking bit_pred CONTENT against BRDC geometry, but that
    /// tested the combiner's published table, NOT what cudaGnssTrack's bit_at() lookup actually
    /// applied -- which is where an indexing error would bite. This closes that gap.
    std::vector<double> pres_c1, pres_c2;
    /// SAME <cos d>, SPLIT by whether the APPLIED head/tail chips agree. This is the off-by-one
    /// test that needs no ground truth. An index shifted by one applies (c[j-1], c[j]) where the
    /// truth is (c[j], c[j+1]), so:
    ///   * applied head == tail  -> the head sign is RIGHT (only the tail can be wrong)
    ///   * applied head != tail  -> the applied head is INVERTED outright
    /// A real off-by-one therefore reads clean on the first and badly degraded on the second.
    /// Any fault that does not care about chip boundaries reads the SAME on both.
    /// pres_fdiff is the mix (fraction of records whose applied pair straddles), ~0.5 on a
    /// balanced overlay and ~0.05 on 20 ms GPS nav data -- which is itself the check that the
    /// pilot/data asymmetry is what the indexing story predicts.
    std::vector<double> pres_same, pres_diff, pres_fdiff;
    /// ...and the ORTHOGONAL split: <cos d> by the VALUE of the applied head chip (+1 vs -1).
    /// The straddle test above conditions on whether NEIGHBOURING chips differ, which is
    /// independent of what the chip IS -- so it cannot see an overlay applied an even/odd
    /// number of times. If the sign is being applied once too many (or once too few), the
    /// residual sign is simply c[j] itself: bimodal 0/pi, flipping on ~50% of records, and
    /// invariant to straddles -- exactly the "pure sign" class (GAL 8, BDS 21/42, L1C 21:
    /// q ~ 0.5, <cos 2d> 0.70-0.84, straddle test flat). Then:
    ///   <cos d | +1> ~ +1 and <cos d | -1> ~ -1  -> the overlay is DOUBLE-APPLIED (or omitted)
    ///   both equal                               -> the sign is not the discriminator
    std::vector<double> pres_pos, pres_neg;
    /// Fraction of gain-measurement records that had NO predicted sign (s_head or s_tail == 0)
    /// and therefore fell back to the DECISION-DIRECTED sgn_obs.
    ///
    /// WHY THIS IS THE ONE THAT MATTERS (2026-07-26). peel_require_bits gates the SUBTRACTION
    /// on have_bits, but the gain MEASUREMENT above runs either way -- without bits it takes
    /// its sign from the running gain. Once that gain is decohered the decision is a coin
    /// flip, which decoheres it further: a self-reinforcing loop whose residual is bimodal
    /// 0/pi and uncorrelated with the overlay, matching the "pure sign" class exactly
    /// (sigma_phi 0.027-0.065 rad on those same sats, so the RAW correlation is pristine and
    /// the flips are manufactured inside the peel).
    /// ⚠️ It is also the blind spot of the two splits above: s_head == 0 lands in NEITHER bin
    /// of the value split and in the "agree" bin of the straddle split, so both read flat no
    /// matter how bad it is. Conditioning on the applied sign cannot see a MISSING one.
    std::vector<double> pres_fzero;
    /// <cos d> split by the TRACKER'S OWN boundary fraction bf = (L - cp_seed)/L at that
    /// record -- the direct test of the TABLE-SEMANTICS theory (2026-07-26):
    ///
    ///   bit_pred's grid is RECORD-INDEXED (table[j] = head-chip of record j; the chip's true
    ///   time span is [start_j-(1-bf)dt, start_j+bf dt]), but bit_at() reads it as
    ///   TIME-DOMAIN ("chip covering time t") -- correct for the GPS LNAV tables whose utc0
    ///   is a true bit edge, WRONG for bit_pred: the head sample t_b - dt/2 lands in the
    ///   previous cell whenever bf < 0.5. The dominant error term is then the NEXT straddle,
    ///   which neither the value split nor the applied-pair straddle split can see -- why
    ///   both read flat. Every other link is now measured-good (content, absolute phase,
    ///   self-consistency, transport, lookup arithmetic), so the semantics is what is left.
    ///
    ///   b<lo>/<hi>: <cos d | bf<0.5> / <cos d | bf>=0.5>.  lo << hi -> confirmed (head-chip
    ///   convention); lo >> hi -> the mirror convention; equal -> theory dead. NB judge on
    ///   pilot chains only; GPS LNAV tables are time-domain and should read equal.
    std::vector<double> pres_blo, pres_bhi;
    /// Why each active PRN is NOT subtracting this record, so the health line can answer it
    /// directly. (The old line reported only warmup lag under "NOT-PEELING", which read as
    /// "nothing is being skipped" while the SNR gate was silently rejecting a third of the sky.)
    enum PeelSkip : uint8_t { PK_PEELING = 0, PK_WARMUP, PK_SNR, PK_NOBITS, PK_AMP };
    std::vector<uint8_t> peel_skip;
    std::vector<double> peel_f_track;   ///< [n_prn] residual carrier (Hz, derotated frame)
    std::vector<double> peel_phi_track; ///< [n_prn] integrated FLL phase (rad, wrapped)
    std::vector<std::complex<double>> peel_a_prev; ///< [n_prn] previous FLL-frame gain measurement
    std::vector<uint8_t> peel_prev_ok;             ///< [n_prn] discriminator history valid
    std::vector<long long> peel_prev_wstart;       ///< [n_prn] sample of the last gain update

    /// TWIN-DESPREAD VERIFIER (peel_verify only). [n_prn] WORST relative prompt error between
    /// the peeled+added-back correlation and a direct despread of the raw window, since the
    /// last health line, and how many records were compared. Worst, not mean: a fault that
    /// only bites on records straddling a code-period boundary would be averaged into silence.
    std::vector<double> peel_verr;
    std::vector<int> peel_vn;
    /// BLOWUP FORENSICS (diagnostic): dump raw per-channel identity terms for this PRN.
    int peel_dbg_prn = -1;
    /// PER-RECORD SIGN GROUND-TRUTH DUMP (2026-07-27). The "pure sign" class -- phase clean
    /// (<cos 2d> ~ 1) but the applied sign wrong on ~half the records, so the gain EMA averages
    /// to zero and a 3000-sigma satellite peels 0 dB -- has resisted all three of the aggregate
    /// splits above (value, straddle, boundary-fraction all read flat or inconsistent over a
    /// full run). Those splits share one blind spot: they score the residual AGAINST the applied
    /// sign, so they cannot see what the sign SHOULD have been.
    ///
    /// This closes that. It streams, per gain-measurement record, exactly what bit_at() APPLIED
    /// plus the absolute time and boundary fraction needed to recompute the truth OFFLINE. For a
    /// PILOT the secondary overlay is fully deterministic, so ground truth needs no decoding and
    /// no broker -- just the PRN, the epoch and the overlay table. Comparing the two columns
    /// names the fault directly (off-by-one, polarity, bf dependence, or none of them) instead
    /// of inferring it from a conditional mean.
    ///
    /// -1 = off (default), 0 = every active PRN on this chain, N = PRN N only.
    /// ⚠️ Writes one line per record per dumped PRN from the GPU stage's host thread: ~100
    /// lines/s/PRN at 10 ms records, ~1000/s at 1 ms. Cheap, but it is real I/O on the
    /// real-time path -- watch kotekan_valve_dropped_frames_total when enabling it, and leave
    /// it off in production.
    int peel_sign_dump_prn = -1;
    std::string peel_sign_dump_path;
    std::unique_ptr<std::ofstream> peel_sign_dump;
    unsigned long long peel_dbg_ctr2 = 0;

    /// What the PREVIOUS frame's peel actually subtracted, so the next gain update can undo it
    /// exactly. Storing it beats recomputing: by update time the EMA has moved on.
    struct PeelUsed {
        uint8_t run = 0;
        int8_t s_head = 0;     ///< predicted bit sign used for the head segment (0 = none:
        int8_t s_tail = 0;     ///< the decision-directed fallback measured this record)
        int job0 = -1;
        double phi = 0.0;      ///< the NCO-mirror phase this record was peeled/measured at
        long long wstart = 0;  ///< window start (sample) -- the FLL's time axis
        float bf = -1.f;       ///< boundary fraction (L - cp_seed)/L at this record, for the
                               ///< table-semantics split (pres_blo/pres_bhi)
    };
    std::vector<PeelUsed> used;                 ///< [MAX_REC][n_prn]
    std::vector<int> used_prn;                  ///< [MAX_REC*n_prn] -> PRN slot, parallel to used
    int used_n_rec = 0;
    std::vector<double2> mir_corr;   ///< host mirror of the previous frame's corr rows
    std::vector<double> mir_energy;  ///< ... and energies
    int mir_rows = 0;
    int mir_rows_spec = 0;
    cudaEvent_t mir_evt = nullptr;   ///< signals the mirror D2H is complete
    bool mir_pending = false;
    long long _peel_diag_ctr = 0; ///< throttles the per-record full-vs-residual diag

    // Device ring bookkeeping (the ring memory itself is a named GPU allocation).
    bool ring_init = false;
    long long hop0 = 0;      ///< absolute hop of ring position 0 (window tiling anchor)
    long long next_hop = 0;  ///< absolute hop the next ingest is expected to start at
    long long next_rec = 0;  ///< next unprocessed record index k (window = hop0 + k*hpr)
};

/**
 * @class cudaGnssTrack
 * @brief cudaProcess command: GNSS channelized tracking despread (phase F0).
 *
 * Per GPU frame: transpose-ingest the staged channelized frame into an internal channel-major
 * device ring (zero-filling valve-drop gaps so ring position == absolute hop - hop0), run
 * pass-1 control for every complete record window (seed extrapolation with the quadratic
 * code-Doppler feed-forward, code-currency translation into the pinned f_ref, fence), then
 * ONE batched E/P/L despread launch per record reading the window in place from the ring.
 * Emits the gnssGpuChain.hpp frame (control block + raw per-channel correlations) into the
 * @c gpu_mem_output array for cudaOutputData; GnssGpuRecordAssemble builds tracker records
 * downstream. Semantics match GnssChannelizedTracker with carrier_shared (fll_gain must be 0).
 *
 * @conf All GnssChannelizedTracker geometry/config keys (signal, sample_rate, f_offset,
 *       spectrum_length, n_channels, channel_offset, num_taps, pfb_window, prns,
 *       hops_per_record, code_doppler_sign, dll_spacing_chips, fll_reacq_hz, capture_utc0,
 *       doppler_margin_hz), plus:
 * @conf gpu_mem_input   staged channelized frame: [n_hops_f][n_chan] complex float, or one
 *                       4+4b byte per sample when quantized_input is set
 * @conf gpu_mem_output  gnssGpuChain frame (frame_bytes(n_prn, n_chan))
 * @conf ring_records    device ring length in records (default 50)
 * @conf seed_endpoint   REST path for broker seeds (default "/track/set_seeds")
 * @conf quantized_input input frames + ring are CHORD 4+4b (GnssQuantize44 upstream; the
 *                       dequantization scales arrive on GnssChanMetadata::chan_scale).
 *                       in_frame_len is then in BYTES = hops * n_channels. Default false.
 */
class cudaGnssTrack : public cudaCommand {
public:
    cudaGnssTrack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                  int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssTrack();

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    cudaGnssTrackState* st();

    /// DIAGNOSTIC (peel_verify): despread the RAW window for the same specs and record, and
    /// accumulate the worst relative prompt error vs the peeled+added-back result into
    /// S.peel_verr. Blocks the stream -- never called in production. See the .cpp note.
    void verify_addback(cudaGnssTrackState& S, const void* d_window, long long wstart,
                        const std::vector<GnssCudaDespread::Spec>& specs,
                        const std::vector<int>& spec_prn, const void* d_corr_peeled,
                        const void* d_scale, cudaStream_t stream);

    std::string _gpu_mem_input, _gpu_mem_output;
    std::string _mem_ring, _mem_jobs, _mem_scale; // stage-namespaced device allocation names
    // Peel scratch, per frame (never leaves the GPU): the float2 residual window, the peel jobs,
    // the gains actually subtracted, and the reference cross-terms the add-back consumes.
    std::string _mem_resid, _mem_pjobs, _mem_gain, _mem_xcorr;
    /// Twin-despread verifier (peel_verify, diagnostic): scratch for a second despread of the
    /// RAW window, so the analytic add-back can be diffed against ground truth in-flight.
    std::string _mem_vcorr, _mem_venergy, _mem_vjobs;
    unsigned long long _verify_ctr = 0; ///< throttle counter for the twin-despread verifier
    int _rows_spec = 4; // gnss_gpu::ROWS_PLAIN, or ROWS_PEEL when this chain peels
    size_t _in_frame_len = 0, _out_frame_len = 0;
    int _n_hops_frame = 0;

    // Host staging for the control block (pageable: the async H2D is host-synchronous on
    // return, so per-instance reuse is safe).
    std::vector<char> _ctl_stage;
    long long _peel_dbg_ctr = 0; ///< throttles the peel INFO log
};

#endif
