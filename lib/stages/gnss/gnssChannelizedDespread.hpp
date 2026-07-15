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
OverlayWipeResult overlay_wipe_at(const std::vector<std::complex<double>>& a,
                                  const std::vector<double>& utc,
                                  const std::vector<int8_t>& overlay, int phase,
                                  const std::vector<std::complex<double>>* head = nullptr);

OverlayWipeResult overlay_wipe(const std::vector<std::complex<double>>& a,
                               const std::vector<double>& utc, const std::vector<int8_t>& overlay,
                               const std::vector<std::complex<double>>* head = nullptr);

} // namespace gnss

#endif // GNSS_CHANNELIZED_DESPREAD_HPP
