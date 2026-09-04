/**
 * @file
 * @brief The two cross-stage code-phase conversions, in ONE place: what the search REPORTS
 *        about a peak, and what the tracker DESPREADS AT given a seed.
 *  - gnss::detection_phase
 *  - gnss::propagate_seed
 *
 * WHY THESE ARE NOT INLINE IN THEIR STAGES ANY MORE (2026-08-02).
 *
 * Every seeding bug found in the week to 2026-08-02 lived in the CONVENTION BETWEEN two
 * stages -- what a number means, what it is reduced modulo, which epoch it references --
 * and not one of them was visible to a test of either stage alone. The arithmetic sat
 * inline in GnssChannelizedSearch::process() and cudaGnssChordTrack::run(), i.e. behind a
 * kotekan buffer graph, a GPU and a REST hop, so the only way to exercise it was to fly it.
 * Bugs were consequently found one at a time, in deployment order, over three days.
 *
 * Pulled out here they are pure functions of numbers, and `scripts/e2e.cpp` can drive the
 * ACTUAL SHIPPED CODE end to end offline in seconds: inject a known code phase, acquire it,
 * report it, seed it, propagate it, despread it, print the error in chips. A harness that
 * re-derives the arithmetic instead of calling it tests the harness author's understanding,
 * which is exactly the thing already known to be unreliable. So: stages call these, the
 * harness calls these, and there is no second copy to drift.
 *
 * @author Keith Vanderlinde
 */

#ifndef GNSS_SEED_TRANSPORT_HPP
#define GNSS_SEED_TRANSPORT_HPP

#include "gnssChannelizedAcquire.hpp" // for AcquisitionSurface
#include "gnssChannelizedReplica.hpp"

#include <complex>
#include <vector>

/// Global-namespace: the GPU despread driver (only built with USE_CUDA). Declared rather
/// than included so this header stays free of the CUDA-side dependency for CPU callers.
class GnssCudaDespread;

namespace gnss {

/// Localize the acquire peak to sub-chip precision by scanning the FULL lag ambiguity.
///
/// WHY THE SPAN IS THE WHOLE STORY (measured 2026-08-02, after two wrong fixes).
///
/// The covering channels share a common factor with @c sph (the aggregator combs 27 channels
/// at stride 4), so the fine-lag response is periodic and carries grating lobes every
/// @c s_stored = 16384/4 = 4096 samples = 13.09 chips. The acquire's own surface is EXACTLY
/// periodic there and cannot choose between them. This despread, however, is not: the lobes
/// differ in height. Dumped over a full hop (`e2e --dump-refine`) on a known-bad injection:
///
///   off  +5376 samp  +17.186 chips   pw 1.5124e-02   <- truth
///   off  +1280 samp   +4.092 chips   pw 1.2818e-02   <- grating lobe, 4096 samples away
///
/// The true lobe wins by ~15%, so ANY scan that reaches it is correct and no tie-breaking is
/// required. The aggregator ran `refine_span: 4096`, could not reach +17.19 chips, and settled
/// on the +4.09 lobe -- an error of exactly 13.09 chips, which is what 4 of 15 noiseless
/// injections showed.
///
/// Two cleverer fixes were tried and are recorded so they are not tried again. Evaluating
/// candidates at exact multiples of @c s_stored fails (13.198 chips out): the true lobe sits at
/// 4096+1280, not at a multiple of 4096, because the coarse estimate carries its own sub-alias
/// error. Making that candidate set symmetric fails worse (26.292 chips out). The lobes are
/// only sph/(comb span in bins) = 157 samples wide, so the step cannot be opened up for a
/// coarse-to-fine pass either -- it would step over the lobe it is hunting. REACH THE PEAK; DO
/// NOT MODEL IT.
///
/// So the span is simply the full ambiguity (default @c fft_len, one whole hop) and the cost is
/// paid with threads: the scan is embarrassingly parallel and @c hoprate_stream is const and
/// touches no shared FFTW state. The reduction is deliberately serial so a tie cannot be broken
/// by thread scheduling -- that is how a deterministic bug learns to look intermittent.
///
/// @param data  the covering channels' columns, in the SAME order as @c cov_global.
/// @param dims  the acquire surface's dimensions. Informational: the scan no longer depends on
///        @c s_stored, but the shape it describes is why the span must be this wide.
/// @param nh_phase  the overlay alignment the peak was found at, or < 0 for none. The refine
///        MUST use it, or it despreads an overlay-blind replica against an overlay-aligned peak
///        and walks off it.
/// @return the refined code phase (the ARGUMENT of a replica at @c anchor), in chips.
double refine_peak(const ChannelizedReplicaBank& bank, int prn_index,
                   const std::vector<std::vector<std::complex<float>>>& data,
                   const std::vector<int>& cov_global, double coarse_cp,
                   const AcquisitionSurface& dims, int nh_phase, double dop, long long anchor,
                   int n_hops, double sample_rate, int refine_span, int refine_step,
                   int n_threads = 1);

#ifdef GNSS_CUDA
/// @ref refine_peak on the GPU (A5 of docs/gnss_gpu_search.md). Same contract, same return.
///
/// With A2/A3 live the refine is 98.2% of a search pass (measured 0.465 s of 0.473 s), because
/// it is `n_eval` exact despreads on the CPU -- the one operation the tracker already does on
/// the GPU. This is a REUSE of GnssCudaDespread, not a new kernel, and it needs no kernel change
/// at all for two reasons that are worth stating because they were not obvious:
///
///  1. THE SCAN IS UNIFORM. Trials sit at `coarse_cp + (-span + e*step)*cps`, and a Spec already
///     emits {cp-ds, cp, cp+ds}. So three consecutive trials are ONE spec with
///     `spacing_chips = step*cps`, and the whole scan is ceil(n_eval/3) specs in one launch.
///  2. THE OVERLAY IS ALREADY IN THE CODE TABLE. GPS_L5_Q_NH declares code_length 204600
///     (= 10230 x 20, NH baked in) and secondary_length 0, so `nh_phase` is INERT for it and the
///     GPU's code lookup carries the overlay for free. The effective code period is therefore
///     20 ms, not 1 ms -- which is why a 391-hop (2.0 ms) refine window contains no code-period
///     boundary at all, and why `m_head` (one boundary) is sufficient for a 10.49 ms record.
///     Computing that window in PRIMARY periods says "2.00 periods, two boundaries" and sends
///     you off to generalize the kernel for nothing. It cost an afternoon; do not repeat it.
///
/// `cp_seed` has the same ARGUMENT convention as @c coarse_cp (build_jobs does
/// `cp0 = comb_mult * cp_seed`, and the kernel forms `C(n) = cp0 + n*cps`), so trial phases pass
/// straight through with no translation.
///
/// One channel group's engine. DespreadJob::chan_mask is a uint64_t, so ONE engine covers at
/// most 64 channels -- ample for the tracker's 7, not for the aggregator's 79-channel union
/// comb. Splitting the refine across groups is EXACT, not an approximation: DespreadResult's
/// `correlation` (the raw coherent sum over channels) and `replica_energy` are both additive,
/// so `amplitude = correlation / replica_energy` recombines from the group totals. Doing it
/// here rather than widening chan_mask keeps the kernel the six live tracker nodes run
/// untouched.
struct CudaRefineGroup {
    ::GnssCudaDespread* gpu = nullptr; ///< engine over exactly this group's channels
    std::vector<int> local;            ///< the group's channel indices in the CALLER's row
    int n_hops = 0;                    ///< hops the engine was built for
    std::vector<std::complex<float>> stage; ///< de-interleave scratch, reused across calls
};

/// @param groups  one entry per <=64-channel group; `stage` is scratch and may be resized
/// @param window  the snapshot, [hop][window_stride] interleaved, as the stage holds it
/// @param window_stride  channels per hop in @c window (the stage's full row, not the group's)
double refine_peak_cuda(std::vector<CudaRefineGroup>& groups, const ChannelizedReplicaBank& bank,
                        int prn_index, const std::complex<float>* window, int window_stride,
                        long long anchor, double coarse_cp, double dop, double sample_rate,
                        int refine_span, int refine_step);
#endif

/// What the search reports about one detected peak's code phase. Three currencies, because
/// three consumers exist and they do not want the same thing:
struct DetectionPhase {
    /// ARGUMENT back-referenced to absolute sample 0, mod primary L. The legacy field
    /// (`code_phase_chips`). Carries a d/d(doppler) lever of ~5095 chips/Hz at CHORD uptime
    /// and cannot express the overlay period. Kept only because the broker's slope fit and
    /// alias census still speak it; NOT what a tracker should be seeded from.
    double cp0 = 0.0;
    /// Same argument, mod n_nh*L, so it can carry which overlay period it is in. -1 when the
    /// signal has no overlay (n_nh <= 1).
    double cp_long = -1.0;
    /// The PHYSICAL code phase at the snapshot's own epoch (ref_hop), mod n_nh*L. This is the
    /// one to transport: its Doppler sensitivity is ~1e-4 chips/Hz, against 5095 for cp0,
    /// because the anchor is a fixed ~1e5 samples rather than the whole 6.8-day uptime.
    double cp_at_ref = -1.0;
};

/// Convert an acquire peak into the reported phases.
///
/// @param bank      the SEARCH's bank -- primary code (GPS_L5_Q, L=10230), overlay OFF in the
///                  table and carried instead as @c best_nh. The long-code reductions below
///                  are done by hand, at n_nh*L, precisely because this bank would reduce them
///                  mod L and destroy the period.
/// @param best_cp   the refined peak: the ARGUMENT of a replica generated at @c anchor that
///                  matches the data at the snapshot.
/// @param best_nh   the overlay alignment the acquire MEASURED (the winning nh_phase), or < 0
///                  if the signal has no overlay.
/// @param n_nh      number of overlay alignments searched (20 for L5 Q5, 1 for none).
/// @param dop       the detection's Doppler (already sign-corrected for the r2c fold).
/// @param snap_start_hop  absolute hop index of the snapshot's first hop.
/// @param anchor    the absolute sample the replicas were generated at (Mp * fft_len).
/// @param sample_rate  full pre-channelization rate, Hz.
/// @param peak_tau_samples  the acquire peak's ABSOLUTE lag (AcquisitionResult::peak_tau_samples).
///        REQUIRED for the overlay period: @c best_cp is the lag already reduced mod one primary
///        period, so the count of whole periods inside the lag -- which spans 16 of them at CHORD
///        and changes every pass -- is only recoverable from here. Omitting it is what made the
///        reported period a pure function of @c best_nh, and wrong 8 times in 15.
DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate, long peak_tau_samples);

/// A seed as it arrives at a tracker (the /set_seeds contract, numeric fields only).
struct SeedState {
    double cp_chips = 0.0;         ///< argument at ref_hop, mod code_length (legacy route)
    double phase_ref_chips = -1.0; ///< PHYSICAL phase at ref_hop; >= 0 wins over cp_chips
    double doppler_hz = 0.0;
    double dop_rate = 0.0; ///< Hz/s
    double cp_rate = 0.0;  ///< RESIDUAL chips/hop (geometry is fed forward here, not by this)
    long long ref_hop = 0;
};

/// What one record's despread is commanded with.
struct SeedPropagation {
    double cp = 0.0;         ///< ARGUMENT to hand the generator for this window
    double doppler_hz = 0.0; ///< Doppler for this window (seed + dop_rate*dt)
    double phase_now = 0.0;  ///< the physical phase it came from (diagnostics)
    double phase_ref = 0.0;  ///< the physical phase at ref_hop (diagnostics)
};

/// Propagate a seed to the record starting at hop @c hop0.
///
/// PROPAGATE IN THE PHASE DOMAIN, NEVER IN THE ARGUMENT DOMAIN (2026-07-31). The despread
/// (like every generator in gnssChannelizedReplica) forms C(n) = cp + n*cps(dop) over the
/// ABSOLUTE sample index, so:
///
///  1. the generator ALREADY applies the full code advance including the nominal 52.3776
///     chips/hop -- adding it here on top (as this stage did between d37064e87 and 9e12d515b)
///     despreads every record ~4969 chips off. airspy's record being exactly one code period
///     made the added term ~0 mod L and hid the double count for the whole prototype era;
///  2. the argument carries the Doppler back-referenced over the entire uptime, ~5095
///     chips/Hz here, and @c dop below moves every record, so a fixed seed argument re-used
///     against a moving Doppler slides thousands of chips per minute.
///
/// So: lift the seed to a physical phase at its own epoch using its OWN Doppler (the lever
/// cancels exactly), propagate the phase -- nominal chips/hop scaled by the code Doppler,
/// plus the residual cp_rate, plus the quadratic dop_rate term -- and convert back with the
/// Doppler this record actually passes to the kernel. A Doppler error then accrues only over
/// dt (seconds) at chip_rate/carrier = 0.0087 chips per Hz per second, which is what the DLL
/// trim is for.
///
/// @param bank   the TRACKER's bank (GPS_L5_Q_NH: overlay baked in, L = 204600), so its own
///               phase_from_arg/arg_from_phase reduce at the length the seed lives at.
/// @param trim_chips  the DLL trim, added ON TOP of the model phase: the broker keeps owning
///               the seed (fit/coast state stays pure) and the trim owns the sub-chip residual.
SeedPropagation propagate_seed(const ChannelizedReplicaBank& bank, const SeedState& sd,
                               long long hop0, double sample_rate, double f_offset_hz,
                               double trim_chips);

} // namespace gnss

#endif // GNSS_SEED_TRANSPORT_HPP
