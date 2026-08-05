/**
 * @file
 * @brief Channelized despread: the measurement-tier operation of the
 *        distributed-band pipeline.
 *
 * In the distributed pipeline a node holds only the PFB channels covering a
 * carrier (see @ref gnssBandPlan.hpp), never the wideband voltage. Once
 * acquisition has supplied a PRN's code phase + Doppler, the calibration
 * product is the complex despread amplitude toward that satellite. This computes
 * it directly in the channelized domain: correlate each covering channel of the
 * data against the *same channel of the replica* (the replica is PFB-analyzed
 * through the identical bank) and sum.
 *
 * Why summing works: the analysis bank is linear and identical for data and
 * replica, so the per-channel cross-correlations carry matching inter-channel
 * phases and add coherently --
 * @f$ G = \sum_c \sum_m X_c[m]\,\overline{R_c[m]} @f$ -- with no need to
 * resynthesize the wideband signal. Normalizing by the replica energy
 * @f$ \sum_c \sum_m |R_c[m]|^2 @f$ yields the matched-filter estimate of the
 * complex amplitude, independent of the bank's passband gain. If the input is
 * exactly @c A times the replica, the estimate is exactly @c A regardless of
 * channelization -- that is the equivalence the unit test pins down.
 */

#ifndef GNSS_CHANNELIZED_DESPREAD_HPP
#define GNSS_CHANNELIZED_DESPREAD_HPP

#include <complex> // for complex
#include <cstdint> // for int8_t
#include <vector>  // for vector

namespace gnss {

/// Result of a channelized despread over a carrier's covering channels.
struct DespreadResult {
    std::complex<double> amplitude;              ///< matched-filter estimate (G / replica_energy)
    std::complex<double> correlation;            ///< raw coherent sum G
    double replica_energy;                       ///< sum_c sum_m |R_c[m]|^2
    std::vector<std::complex<double>> per_channel; ///< g_c per provided channel
};

/**
 * Coherent channelized despread. @c data_ch and @c repl_ch are parallel lists of
 * the *same* channels (e.g. those from @ref covering_channels), each a per-hop
 * stream of channelizer outputs; corresponding streams must be equal length.
 * Returns the per-channel correlations, their coherent sum, the replica energy,
 * and the normalized amplitude estimate (0 if the replica has no energy).
 */
DespreadResult channelized_despread(const std::vector<std::vector<std::complex<float>>>& data_ch,
                                    const std::vector<std::vector<std::complex<float>>>& repl_ch);

/// Result of a known-secondary-overlay deep wipe (@ref overlay_wipe).
struct OverlayWipeResult {
    double amplitude = 0.0; ///< deep coherent |A| = |sum of overlay-corrected per-record A| / nrec
    double snr = 0.0;       ///< significance = coherent sum / its orthogonal-noise std (~1 noise, >>1 real)
    int phase = 0;          ///< overlay alignment (0..len-1) that maximised the coherent sum
};

/**
 * Deep coherent integration of a dataless pilot past its primary period by wiping a KNOWN
 * secondary overlay (the L5 Neuman-Hofman NH10/NH20 -- one +-1 chip per primary period).
 *
 * @c a is the per-record complex despread amplitude (one record = one primary period, so one
 * overlay chip), @c utc the matching capture time per record (to index the ABSOLUTE primary-
 * period -- a dropped record just skips an index, keeping the overlay aligned), @c overlay the
 * bipolar (+-1) sequence. Unlike the nav-bit wipe (estimates the +-1 by squaring), the overlay
 * is KNOWN, so this just searches its @c overlay.size() alignments for the one that maximises
 * the coherent sum, then sums the overlay-corrected records -- recovering the pilot's full
 * coherent gain (capped only by the carrier coherence time, not the 1 ms primary period).
 */
/// Plain coherent sum of per-record amplitudes -- NO wipe of any kind.
///
/// For a pilot whose replica ALREADY carries the secondary overlay there is nothing left to
/// wipe: the despread removed it chip by chip, correctly, whether or not a record straddles an
/// overlay transition. CHORD is exactly that case -- its tracker despreads `GPS_L5_Q_NH`, the
/// 204600-chip code with NH20 baked in -- and it CANNOT use the wipe rungs, because
/// @ref overlay_apply advances the overlay ONE CHIP PER RECORD. That identity holds on airspy
/// (record = one primary period) and fails on CHORD, where a record is 2048 hops = **10.4857**
/// code periods, so the overlay would advance ~10.49 chips per record and flip ~10 times WITHIN
/// one -- which the head/tail split cannot represent.
///
/// Without this, such a signal has no deep-integration path at all: every route to
/// `coherence_s` in GnssCoherentCombiner runs through a wipe, so the stage reports 0 forever
/// and the pilot is integrated one record at a time (10.5 ms here) when it supports far longer.
///
/// Same SNR convention as @ref overlay_wipe_at -- coherent sum over its own orthogonal-noise
/// std, E[snr^2] = 2 exactly under pure noise -- so the result is directly comparable with the
/// other rungs and their floors. @c phase is always 0: there is no alignment to report, and
/// (the point) no max-over-alignments, so the noise floor carries no extreme-value inflation.
OverlayWipeResult coherent_sum(const std::vector<std::complex<double>>& a);

/// LEAVE-ONE-OUT COMMON-PHASE TRACKER -- the closed carrier loop the airspy had, in batch form.
///
/// The on-sky per-record prompts carry a slow (~>42 ms correlated) per-satellite phase wander
/// that is NOT a polynomial of low order (measured ~0.9 rad rms over 1 s, red spectrum, common
/// across every antenna and subband -- propagation/sky, not processing). The airspy never saw
/// it because its per-record phase estimate was signal-dominated and its carrier loop TRACKED
/// it out before any long integration; CHORD's open-loop polynomial derotation cannot. This is
/// the batch equivalent of closing that loop: estimate each record's common phase from its
/// NEIGHBOURS within the wander's correlation time and derotate.
///
/// Each record k is derotated by arg(sum of a[j], j in [k-half_width, k+half_width], j != k,
/// zero-amplitude records skipped). SELF-EXCLUDED, deliberately: the estimate is then
/// statistically independent of record k's own noise, so under pure noise the derotation is a
/// random rotation that cannot align anything -- the tracker is FAIL-CLOSED (it cannot
/// manufacture a detection) and coherent_sum's E[snr^2] = 2 noise convention survives it.
/// The price is estimator noise ~1/(SNR_rec * sqrt(2*half_width)) rad of added jitter, so
/// tracking a clean strong signal costs ~exp(-sigma_est^2/2) -- a few percent -- while removing
/// a 0.9 rad wander recovers ~exp(0.9^2/2) ~ 1.5x amplitude. Callers should score tracked and
/// untracked with the SAME estimator and keep the better (a 2-way selection the floor must pay).
///
/// Records with fewer than 2 usable neighbours pass through unrotated. Returns false (out_a
/// untouched) on a degenerate window (< 4 records or half_width < 1).
bool phase_track_loo(const std::vector<std::complex<double>>& a, int half_width,
                     std::vector<std::complex<double>>& out_a);

/// Selection-free overlay wipe at a GIVEN alignment (dead-reckoned by the caller):
/// one coherent sum, no phase search -- E[snr^2] = 2 exactly under noise.
///
/// SEGMENTED wipe (@c head non-null): records are hop-aligned, not code-period-aligned, so
/// each one straddles a period boundary where the overlay flips -- summed blind, a record
/// straddling a chip TRANSITION cancels to |2f-1| (the 2026-07-15 "bistable": f~0.5 nulled
/// 12/25 E1C records / ~49% of B1C). @c head[r] is the record's correlation over the hops
/// BEFORE its boundary (tail = a[r] - head[r]); the wipe applies chip k to the head and
/// chip k+1 to the tail, so nothing ever cancels. head == a (tail 0) reduces exactly to
/// the unsegmented behaviour.
/// Apply a KNOWN overlay at a GIVEN alignment and hand back the wiped per-record amplitudes,
/// without reducing them to a coherent sum. This exists for the signal no other chain is:
/// a DATA component that ALSO carries a secondary overlay (GPS L5-I = CNAV under NH10). Such a
/// chain needs BOTH wipes in order -- overlay first, then the nav-bit wipe on the result -- and
/// the two were mutually exclusive branches until S4 phase 2.
///
/// @p out_head (optional) receives the wiped HEAD segment of each record, so a downstream
/// SEGMENTED wipe keeps a consistent head/tail split (its tail is out_a - out_head).
///
/// ★ This is THE ONE PLACE the overlay's per-record chip index is computed; overlay_wipe_at()
/// calls it rather than repeating the expression. A second, silently-divergent copy of an index
/// convention is exactly what produced the L5 "pure sign" peel fault (2026-07-27).
///
/// @returns false if the window is degenerate (too short, size mismatch, no positive record
/// period), in which case @p out_a is not usable.
bool overlay_apply(const std::vector<std::complex<double>>& a, const std::vector<double>& utc,
                   const std::vector<int8_t>& overlay, int phase,
                   const std::vector<std::complex<double>>* head,
                   std::vector<std::complex<double>>& out_a,
                   std::vector<std::complex<double>>* out_head);

OverlayWipeResult overlay_wipe_at(const std::vector<std::complex<double>>& a,
                                  const std::vector<double>& utc,
                                  const std::vector<int8_t>& overlay, int phase,
                                  const std::vector<std::complex<double>>* head = nullptr);

OverlayWipeResult overlay_wipe(const std::vector<std::complex<double>>& a,
                               const std::vector<double>& utc, const std::vector<int8_t>& overlay,
                               const std::vector<std::complex<double>>* head = nullptr);

} // namespace gnss

#endif // GNSS_CHANNELIZED_DESPREAD_HPP
