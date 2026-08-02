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

#include "gnssChannelizedReplica.hpp"

namespace gnss {

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
DetectionPhase detection_phase(const ChannelizedReplicaBank& bank, double best_cp, int best_nh,
                               int n_nh, double dop, long long snap_start_hop, long long anchor,
                               double sample_rate);

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
