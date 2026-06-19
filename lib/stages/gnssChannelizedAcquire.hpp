/**
 * @file
 * @brief Channelized acquisition: find a GNSS signal's code phase + Doppler
 *        from PFB channels alone, with no wideband resynthesis.
 *
 * The measurement tier (@ref gnssChannelizedDespread.hpp) needs a seed code
 * phase + Doppler. This produces them from the same channelized voltages a node
 * already holds. The key is recovering *full* code-phase resolution: each
 * channel is decimated by N, so a per-channel correlation resolves the lag only
 * to a hop (N full-rate samples) -- far coarser than one chip. Writing the lag
 * as @f$ \tau = qN + s @f$, the sub-hop part @c s appears as a phase ramp across
 * channels, so
 * @f[ D(qN+s) = \sum_c P_c(q)\,e^{+i 2\pi c s / N}, \qquad
 *     P_c(q) = \sum_m X_c[m]\,\overline{R_c[m-q]} . @f]
 * The coarse lag @c q comes from a per-channel circular correlation; the fine
 * lag @c s from an N-point DFT of @c {P_c(q)} across channels. That cross-channel
 * DFT is the only "reassembly" -- a tiny transform of correlation values, never
 * of the signal -- so acquisition stays entirely local to the channels held.
 * Doppler is wiped off the data as a per-hop phase (GNSS Doppler << channel
 * bandwidth) and scanned over a grid.
 *
 * The reconstruction is exact for a critically-sampled rectangular bank and
 * approximate for the windowed PFB; that is sufficient because acquisition only
 * has to localize the peak to ~1 chip, after which the exact despread measures.
 * Channel/lag conventions match @ref pfbPrototype.hpp analysis with no fftshift
 * (channel c <-> +frequency c); a stage feeding fftwEngine output must remap the
 * fftshift accordingly.
 */

#ifndef GNSS_CHANNELIZED_ACQUIRE_HPP
#define GNSS_CHANNELIZED_ACQUIRE_HPP

#include <complex> // for complex
#include <vector>  // for vector

namespace gnss {

/// Outcome of a channelized acquisition search.
struct AcquisitionResult {
    double code_phase_chips; ///< peak code phase, chips
    double doppler_hz;       ///< peak Doppler from the grid, Hz
    long peak_tau_samples;   ///< peak lag, full-rate samples (= q*N + s)
    double peak;             ///< peak |D|^2
    double snr;              ///< peak / mean of the |D|^2 surface
};

/**
 * Search the (code phase, Doppler) surface from channelized voltages.
 *
 * @param data_ch    Channelized data, [N][M] (N channels, M hops).
 * @param repl0_ch   Code-only replica (code phase 0, Doppler 0), analyzed through
 *                   the identical bank, [N][M].
 * @param covering   Channel indices carrying the carrier (others are noise-only
 *                   and excluded from the coherent sum).
 * @param doppler_grid Trial Doppler frequencies, Hz.
 * @param sample_rate  Full (pre-channelization) sample rate, Hz.
 * @param chip_rate    Spreading code chip rate, Hz.
 * @param num_chan     N, the channel count (= decimation = hop size in samples).
 * @param code_length  Chips per code period (for wrapping the code phase).
 * @param chan_freq   Optional integer frequency index of each covering channel,
 *                    used in the cross-channel fine-lag ramp. Empty defaults to
 *                    `covering` (channel c <-> +frequency c, the clean bank). A
 *                    fftshifted bank (fftwEngine) passes c - N/2. The per-channel
 *                    analysis phase cancels in P_c, so only this index matters.
 */
AcquisitionResult
channelized_acquire(const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                    const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                    double sample_rate, double chip_rate, int num_chan, long code_length,
                    const std::vector<int>& chan_freq = {});

} // namespace gnss

#endif // GNSS_CHANNELIZED_ACQUIRE_HPP
