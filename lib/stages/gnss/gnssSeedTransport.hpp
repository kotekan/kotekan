/**
 * @file
 * @brief The cross-stage code-phase conversions: what the search reports about a peak, and
 *        what the tracker despreads at given a seed.
 *  - gnss::refine_peak
 *  - gnss::detection_phase
 *  - gnss::propagate_seed
 *
 * These encode the conventions between stages (what a code phase means, what it is reduced
 * modulo, which epoch it refers to), which no test of a single stage can check. As pure
 * functions they are called both by the stages and by the offline harness
 * scripts/gnss/e2e.cpp, which injects a known code phase, acquires, reports, seeds, propagates
 * and despreads it, and prints the error in chips. There is no second copy to drift.
 *
 * @author Keith Vanderlinde
 */

#ifndef GNSS_SEED_TRANSPORT_HPP
#define GNSS_SEED_TRANSPORT_HPP

#include "gnssChannelizedReplica.hpp"

#include <complex>
#include <vector>

/// The GPU despread driver (global namespace, CUDA builds only). Declared rather than included
/// so CPU callers do not depend on the CUDA headers.
class GnssCudaDespread;

namespace gnss {

/// Localize the acquire peak to sub-chip precision by scanning the whole lag ambiguity.
///
/// When the covering channels share a common factor with the samples per hop (a comb at
/// stride 4, say), the fine-lag response is periodic, with grating lobes every @c s_stored
/// samples (4096 samples = 13.09 chips for L5 at CHORD). The acquire surface is exactly
/// periodic there and cannot choose between them; this exact despread can, because the true
/// lobe is the highest. A scan that reaches the true lobe therefore picks it, and one that
/// stops short settles on a grating lobe a multiple of @c s_stored away. Testing only
/// candidates at multiples of @c s_stored does not work, because the coarse estimate has its
/// own sub-alias error, and the lobes are only ~sph/(comb span in bins) samples wide, so a
/// coarse step would skip over them.
///
/// The scan therefore covers +-@c refine_span samples, which must reach across the whole
/// ambiguity (the stage default is @c fft_len, one hop). It is spread over @c n_threads:
/// @c hoprate_stream_into is const and touches no shared FFTW state. The reduction is serial
/// (ascending, first strict maximum), so a tie never depends on thread scheduling.
///
/// @param data  the covering channels' columns, [chan][hop], in the same order as
///        @c cov_global; only the leading @c n_hops of each are used.
/// @param nh_phase  the overlay alignment the peak was found at, or < 0 for none. Without it
///        the refine despreads an overlay-blind replica against an overlay-aligned peak.
/// @return the refined code phase (the argument of a replica at @c anchor), in chips.
double refine_peak(const ChannelizedReplicaBank& bank, int prn_index,
                   const std::vector<std::vector<std::complex<float>>>& data,
                   const std::vector<int>& cov_global, double coarse_cp, int nh_phase, double dop,
                   long long anchor, int n_hops, double sample_rate, int refine_span,
                   int refine_step, int n_threads = 1);

#ifdef GNSS_CUDA
/// One channel group's GPU despread engine for @ref refine_peak_cuda. DespreadJob::chan_mask is
/// a uint64_t, so one engine covers at most 64 channels. Splitting the refine across groups is
/// exact: DespreadResult's `correlation` (the coherent sum over channels) and `replica_energy`
/// both add over channels, so `amplitude = correlation / replica_energy` recombines from the
/// group totals.
struct CudaRefineGroup {
    ::GnssCudaDespread* gpu = nullptr;      ///< engine over exactly this group's channels
    std::vector<int> local;                 ///< the group's channel indices in the caller's row
    int n_hops = 0;                         ///< hops the engine was built for
    std::vector<std::complex<float>> stage; ///< de-interleave scratch, reused across calls
};

/// @ref refine_peak on the GPU, through GnssCudaDespread: the same scan, tie-break and return.
///
/// Trials sit at `coarse_cp + (-span + e*step)*cps` and a Spec despreads at
/// {cp - spacing, cp, cp + spacing}, so three consecutive trials are one Spec with
/// `spacing_chips = step*cps`; the scan is ceil(n_eval/3) Specs, chunked to
/// GnssCudaDespread::max_batch_specs. `cp_seed` is an argument in the same convention as
/// @c coarse_cp (the kernel forms C(n) = comb_mult*cp_seed + n*cps), so trial phases pass
/// through unchanged.
///
/// The GPU despread takes no nh_phase. A signal with its overlay tiled into the code table
/// (GPS_L5_Q_NH) carries it through the code lookup; a bank with a per-period overlay
/// (secondary_length > 0) is scanned with an overlay-blind replica, unlike @ref refine_peak.
///
/// @param groups  one entry per <=64-channel group; `stage` is scratch and may be resized
/// @param window  the snapshot, [hop][window_stride] interleaved, as the stage holds it
/// @param window_stride  channels per hop in @c window (the stage's full row, not the group's)
double refine_peak_cuda(std::vector<CudaRefineGroup>& groups, const ChannelizedReplicaBank& bank,
                        int prn_index, const std::complex<float>* window, int window_stride,
                        long long anchor, double coarse_cp, double dop, double sample_rate,
                        int refine_span, int refine_step);
#endif

/// What the search reports about one detected peak's code phase, in three forms for three
/// kinds of consumer.
struct DetectionPhase {
    /// Argument referred back to absolute sample 0, mod the primary length L (published as
    /// `code_phase_chips`): the argument the data would have been generated with at the
    /// detection's Doppler, so re-advancing it to any sample at that Doppler gives the
    /// physical phase there (the broker's _det_offset). As an argument it carries a
    /// d/d(doppler) lever of ~5095 chips/Hz at 6.8 days of uptime, which cancels only when
    /// it is re-advanced with the Doppler it was published with, and it cannot express the
    /// overlay period. A tracker should be seeded from @c cp_at_ref.
    double cp0 = 0.0;
    /// The same argument mod n_nh*L, so it carries the overlay period. -1 when the signal has
    /// no overlay (n_nh <= 1 or best_nh < 0).
    double cp_long = -1.0;
    /// Physical code phase at the last sample of the snapshot's first hop (ref_hop), mod
    /// n_nh*L. This is the one to transport: it is referred to its own epoch rather than to
    /// sample 0, so its Doppler sensitivity is ~1e-4 chips/Hz. -1 when the signal has no
    /// overlay, as for @c cp_long. At every Doppler, cp_at_ref mod L equals cp0 re-advanced
    /// to the same sample (ChannelizedReplicaBank::phase_from_arg).
    double cp_at_ref = -1.0;
};

/// Convert an acquire peak into the reported phases.
///
/// @param bank      the search's bank: primary code only (e.g. GPS_L5_Q, L=10230), with the
///                  overlay carried as @c best_nh. The long-code reductions are done here at
///                  n_nh*L, because the bank's own conversions reduce mod L and lose the period.
/// @param best_cp   the refined peak: the argument of a replica generated at @c anchor that
///                  matches the data at the snapshot. The replica's advance from sample 0 to
///                  @c anchor is part of that argument, so every reported argument removes
///                  the advance over snap_start_hop*fft_len - anchor, never over the snapshot
///                  start alone (the anchor's code-Doppler share was #207).
/// @param best_nh   the overlay alignment the acquire measured (the winning nh_phase), or < 0
///                  if the signal has no overlay.
/// @param n_nh      number of overlay alignments searched (20 for L5 Q5, 1 for none).
/// @param dop       the detection's Doppler (already sign-corrected for the r2c fold).
/// @param snap_start_hop  absolute hop index of the snapshot's first hop.
/// @param anchor    the absolute sample the replicas were generated at (Mp * fft_len).
/// @param sample_rate  full pre-channelization rate, Hz.
/// @param peak_tau_samples  the acquire peak's full lag (AcquisitionResult::peak_tau_samples).
///        Needed for the overlay period: @c best_cp is the lag reduced mod one primary period,
///        and the count of whole periods inside the lag (it spans 16 at CHORD, and changes from
///        pass to pass) can only be recovered from here.
DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate, long peak_tau_samples);

/// A seed as it arrives at a tracker (the /set_seeds contract, numeric fields only).
struct SeedState {
    double cp_chips = 0.0;         ///< argument at ref_hop, mod code_length
    double phase_ref_chips = -1.0; ///< physical phase at ref_hop, component chips; >= 0 wins
    double doppler_hz = 0.0;
    double dop_rate = 0.0; ///< Hz/s
    double cp_rate = 0.0;  ///< residual chips/hop; the geometric code Doppler is added separately
    long long ref_hop = 0;
};

/// What one record's despread is commanded with.
struct SeedPropagation {
    double cp = 0.0;         ///< argument to hand the generator for this window
    double doppler_hz = 0.0; ///< Doppler for this window (seed + dop_rate*dt)
    double phase_now = 0.0;  ///< the physical phase it came from, combined chips (diagnostics)
    double phase_ref = 0.0;  ///< the physical phase at ref_hop, combined chips (diagnostics)
};

/// Propagate a seed to the record starting at hop @c hop0.
///
/// Propagation is in the phase domain, never the argument domain. The despread, like every
/// generator in gnssChannelizedReplica, forms C(n) = comb_mult*cp + n*cps(dop) over the
/// absolute sample index, so:
///
///  1. the generator already applies the full code advance, including the nominal chips per
///     hop; adding it to the argument as well counts it twice;
///  2. the argument carries the Doppler lever over the whole uptime (~5095 chips/Hz at 6.8
///     days), and the Doppler changes every record, so a fixed argument used against a
///     moving Doppler slides thousands of chips per minute.
///
/// So the seed is lifted to a physical phase at its own epoch with its own Doppler (the lever
/// cancels exactly), propagated as a phase (nominal chips/hop scaled by the code Doppler, plus
/// the residual cp_rate, plus the dop_rate quadratic term), and converted back with the
/// Doppler this record passes to the kernel. A Doppler error then accrues only over dt, at
/// chip_rate/carrier chips per Hz per second (0.0087 for L5), which the DLL trim absorbs.
///
/// @param bank   the tracker's bank, whose phase_from_arg/arg_from_phase reduce at the code
///               length the seed lives at (204600 for GPS_L5_Q_NH, overlay tiled in).
/// @param f_offset_hz  divides the dop_rate term's code Doppler (chip_rate / f_offset_hz);
///               callers pass the stage's f_offset_hz, the carrier itself in CHORD configs.
/// @param trim_chips  the DLL trim, added unscaled to the propagated (combined-chip) phase. The
///               broker's seed stays the model and the trim carries the sub-chip residual.
SeedPropagation propagate_seed(const ChannelizedReplicaBank& bank, const SeedState& sd,
                               long long hop0, double sample_rate, double f_offset_hz,
                               double trim_chips);

} // namespace gnss

#endif // GNSS_SEED_TRANSPORT_HPP
