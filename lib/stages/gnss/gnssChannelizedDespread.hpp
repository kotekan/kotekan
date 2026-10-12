/**
 * @file
 * @brief Channelized despread, and the deep coherent integrations over its per-record
 *        amplitudes (overlay wipes, plain coherent sum, leave-one-out phase tracker).
 *
 * A node holds only the PFB channels covering a carrier (see @ref gnssBandPlan.hpp), never
 * the wideband voltage. Given a PRN's code phase and Doppler, the despread correlates each
 * covering channel of the data against the same channel of the replica, which went through
 * the same analysis bank, and sums.
 *
 * The bank is linear and identical for data and replica, so the per-channel correlations
 * carry matching inter-channel phases and add coherently,
 * @f$ G = \sum_c \sum_m X_c[m]\,\overline{R_c[m]} @f$, without resynthesizing the wideband
 * signal. Normalizing by the replica energy @f$ \sum_c \sum_m |R_c[m]|^2 @f$ gives the
 * matched-filter estimate of the complex amplitude, independent of the bank's passband gain:
 * an input of exactly @c A times the replica gives exactly @c A (pinned by
 * test_gnss_channelized_despread.cpp).
 */

#ifndef GNSS_CHANNELIZED_DESPREAD_HPP
#define GNSS_CHANNELIZED_DESPREAD_HPP

#include <complex> // for complex
#include <cstdint> // for int8_t
#include <vector>  // for vector

namespace gnss {

/// Result of a channelized despread over a carrier's covering channels.
struct DespreadResult {
    std::complex<double> amplitude;                ///< matched-filter estimate (G / replica_energy)
    std::complex<double> correlation;              ///< raw coherent sum G
    double replica_energy;                         ///< sum_c sum_m |R_c[m]|^2
    std::vector<std::complex<double>> per_channel; ///< g_c per provided channel
};

/**
 * Coherent channelized despread. @c data_ch and @c repl_ch are parallel lists of the same
 * channels (e.g. those from @ref covering_channels), each a per-hop stream of channelizer
 * outputs. Returns the per-channel correlations, their coherent sum, the replica energy, and
 * the normalized amplitude estimate (0 if the replica has no energy). Throws
 * std::invalid_argument if the channel counts or a pair of stream lengths differ.
 */
DespreadResult channelized_despread(const std::vector<std::vector<std::complex<float>>>& data_ch,
                                    const std::vector<std::vector<std::complex<float>>>& repl_ch);

/// Result of @ref overlay_wipe, @ref overlay_wipe_at and @ref coherent_sum.
struct OverlayWipeResult {
    double amplitude = 0.0; ///< deep coherent |A| = |sum of overlay-corrected per-record A| / nrec
    double snr = 0.0;       ///< coherent sum / its orthogonal-noise std (order 1 on noise)
    /// Overlay alignment (0..len-1), relative to the first record: the maximising one from
    /// overlay_wipe, the given one from overlay_wipe_at, 0 from coherent_sum.
    int phase = 0;
};

/// Plain coherent sum of per-record amplitudes, with no wipe.
///
/// For a pilot whose replica already carries the secondary overlay (CHORD despreads
/// GPS_L5_Q_NH, the 204600-chip code with NH20 tiled in) there is nothing left to wipe. The
/// wipes below cannot serve it anyway: @ref overlay_apply advances the overlay one chip per
/// record and its head/tail split allows one chip change within a record, while a CHORD
/// record (2048 hops, 10.49 ms) spans ~10.49 primary periods.
///
/// Same SNR convention as @ref overlay_wipe_at, so the result is comparable with the wipes'
/// floors. @c phase is always 0.
OverlayWipeResult coherent_sum(const std::vector<std::complex<double>>& a);

/// Leave-one-out common-phase tracker: a closed carrier loop in batch form, for a per-record
/// phase wander that is correlated across the +-half_width window but is not a low-order
/// polynomial.
///
/// Each record k is derotated by arg(sum of a[j] for j in [k-half_width, k+half_width],
/// j != k, zero-amplitude records skipped). Excluding record k makes the estimate independent
/// of its own noise, so on pure noise each derotation is a random rotation of that record.
/// Records still enter their neighbours' estimates, though, so the tracked sum's noise SNR has
/// a heavier tail than the untracked one's, more so as half_width grows. On a signal the cost
/// is ~1/(SNR_rec * sqrt(2*half_width)) rad of added jitter, a loss of ~exp(-sigma_est^2/2)
/// when there is no wander. Callers should score tracked and untracked candidates with the
/// same estimator and keep the better, paying for that selection in the floor.
///
/// Records with fewer than 2 usable neighbours pass through unrotated. Returns false (out_a
/// untouched) on a degenerate window (< 4 records or half_width < 1).
bool phase_track_loo(const std::vector<std::complex<double>>& a, int half_width,
                     std::vector<std::complex<double>>& out_a);

/// Apply a known overlay at a given alignment and return the wiped per-record amplitudes, for
/// a data component under an overlay (GPS L5-I: CNAV under NH10) that needs the nav-bit wipe
/// applied after the overlay wipe.
///
/// Record r takes overlay chip k = round((utc[r]-utc[0])/rec_dt) + @p phase (mod the overlay
/// length), with rec_dt the median record spacing, so @p phase is relative to the first record
/// and a dropped record skips an index rather than shifting the rest. overlay_wipe_at() wipes
/// through this function; overlay_wipe() computes the same index itself, and the two must
/// agree.
///
/// @p head (optional) segments the wipe as in @ref overlay_wipe_at, and is ignored if its size
/// differs from @p a. @p out_head (optional) receives each record's wiped head, so a
/// downstream segmented wipe keeps a consistent head/tail split.
///
/// @returns false if the window is degenerate (fewer than 2 records, utc size mismatch, empty
/// overlay, or no positive record period), in which case @p out_a is not usable.
bool overlay_apply(const std::vector<std::complex<double>>& a, const std::vector<double>& utc,
                   const std::vector<int8_t>& overlay, int phase,
                   const std::vector<std::complex<double>>* head,
                   std::vector<std::complex<double>>& out_a,
                   std::vector<std::complex<double>>* out_head);

/// Overlay wipe at a given alignment @p phase (see @ref overlay_apply): one coherent sum with no
/// search over alignments, so E[snr^2] = 2n/(n-3) ~ 2 for n records of pure noise, with no
/// extreme-value inflation.
///
/// With @p head non-null the wipe is segmented. Records are hop-aligned, not period-aligned,
/// so a record straddles a period boundary where the overlay may flip; summed whole, a record
/// straddling a chip transition at fraction f cancels to |2f-1|. @c head[r] is the record's
/// correlation over the hops before its boundary; the wipe applies chip k to the head and chip
/// k+1 to the tail a[r] - head[r]. head == a reduces to the unsegmented wipe.
OverlayWipeResult overlay_wipe_at(const std::vector<std::complex<double>>& a,
                                  const std::vector<double>& utc,
                                  const std::vector<int8_t>& overlay, int phase,
                                  const std::vector<std::complex<double>>* head = nullptr);

/**
 * Deep coherent integration of a dataless pilot past its primary period by wiping a known
 * secondary overlay, shared (L5 Q's NH20) or per-PRN (L1C-P's). Each record must span one
 * primary period, so the overlay advances one +-1 chip per record.
 *
 * @c a is the per-record complex despread amplitude, @c utc the capture time of each record
 * (the record index comes from it, so a dropped record skips an index and the overlay stays
 * aligned), @c overlay the bipolar sequence, @c head as for @ref overlay_wipe_at. Unlike
 * the nav-bit wipe, which estimates each +-1 symbol, the overlay is known, so this searches its
 * @c overlay.size() alignments for the one that maximises the coherent sum. The returned SNR
 * carries that selection: on noise it rectifies to ~sqrt(2 ln len).
 */
OverlayWipeResult overlay_wipe(const std::vector<std::complex<double>>& a,
                               const std::vector<double>& utc, const std::vector<int8_t>& overlay,
                               const std::vector<std::complex<double>>* head = nullptr);

} // namespace gnss

#endif // GNSS_CHANNELIZED_DESPREAD_HPP
