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

} // namespace gnss

#endif // GNSS_CHANNELIZED_DESPREAD_HPP
