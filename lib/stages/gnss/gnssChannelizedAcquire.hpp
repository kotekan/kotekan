/**
 * @file
 * @brief Channelized acquisition: find a GNSS signal's code phase and Doppler from PFB
 *        channels alone, with no wideband resynthesis.
 *
 * Produces the seed code phase and Doppler for the measurement tier
 * (@ref gnssChannelizedDespread.hpp) from the channelized voltages a node already holds. Each
 * channel is decimated by @c sph (full-rate samples per hop), so a per-channel correlation
 * resolves the lag only to a hop, far coarser than a chip. Writing the lag as
 * @f$ \tau = q\,\mathrm{sph} + s @f$, the sub-hop part @c s appears as a phase ramp across
 * channels:
 * @f[ D(q\,\mathrm{sph}+s) = \sum_c P_c(q)\,e^{+i 2\pi c s / \mathrm{sph}}, \qquad
 *     P_c(q) = \sum_m X_c[m]\,\overline{R_c[m-q]} . @f]
 * The coarse lag @c q comes from a per-channel circular correlation and the fine lag @c s from
 * an sph-point DFT of @c {P_c(q)} across channels. The circular correlation wraps the replica
 * at its hop period @c Mp (the length of @c repl0_ch), so the data window @c M may differ from
 * @c Mp and a code period may be a fractional number of hops (195.3125 hops per 1 ms period at
 * CHORD, where Mp = 3125 hops spans 16 periods). The cross-channel DFT is a small transform of
 * correlation values, never of the signal, so acquisition stays local to the channels held.
 * Doppler is wiped off the data as a per-hop phase (GNSS Doppler << channel bandwidth) and
 * scanned over a grid.
 *
 * The reconstruction is exact for a critically sampled rectangular bank and approximate for a
 * windowed PFB. That is sufficient because acquisition only has to localize the peak to ~1
 * chip; the exact despread measures after it. Channel c is taken to be at frequency index c
 * unless @c chan_freq says otherwise (natural order, as an r2c bank emits; an fftshifted bank
 * passes its own indices). The bank must fold newest-first, as the CHORD F-engine does: the fold
 * order sets the direction of the fine-lag axis (FINE_LAG_SIGN_PFB).
 */

#ifndef GNSS_CHANNELIZED_ACQUIRE_HPP
#define GNSS_CHANNELIZED_ACQUIRE_HPP

#include <complex> // for complex
#include <fftw3.h> // for fftwf_plan, fftwf_complex
#include <vector>  // for vector

namespace gnss {

/**
 * Reusable FFTW workspace for the coarse correlation in @ref channel_correlate: length-@c Mp
 * complex plans and scratch buffers. Sized on first use and re-planned only when @c Mp
 * changes, so create one and reuse it across windows. Non-copyable; owns its FFTW resources.
 * Plan creation and destruction take the shared FFTW planner lock; execution does not.
 */
struct AcquireWorkspace {
    AcquireWorkspace() = default;
    ~AcquireWorkspace();
    AcquireWorkspace(const AcquireWorkspace&) = delete;
    AcquireWorkspace& operator=(const AcquireWorkspace&) = delete;
    /// (Re)allocate plans and buffers for length-@c n circular transforms.
    void ensure(int n);

    int Mp = 0;                   ///< current transform length (0 = unallocated)
    fftwf_complex* in = nullptr;  ///< scratch input  (length Mp)
    fftwf_complex* out = nullptr; ///< scratch output (length Mp)
    fftwf_plan fwd = nullptr;     ///< length-Mp c2c forward  (in -> out)
    fftwf_plan inv = nullptr;     ///< length-Mp c2c backward (in -> out, unnormalized)
};

/// Outcome of a channelized acquisition search.
struct AcquisitionResult {
    double code_phase_chips; ///< peak code phase, chips
    double doppler_hz;       ///< peak Doppler, Hz, refined between grid points where possible
    long peak_tau_samples;   ///< peak lag, full-rate samples, modulo Mp*sph
    double peak;             ///< peak |D|^2 (the raw grid sample)
    double snr;              ///< peak / mean of the |D|^2 surface
    /// Scalloping-corrected peak: the mainlobe pair-sum dop_peak[d] + max(dop_peak[d +- 1]) at
    /// the chosen Doppler plane d. Equals @c peak when d is at a grid edge or no per-plane
    /// maxima were given.
    ///
    /// Use this, not @c peak or @c snr, to compare hypotheses (e.g. overlay alignments) whose
    /// Doppler is a nuisance parameter: on a critically sampled grid the true peak scallops to
    /// 0.405 of its power at mid-bin, while an overlay-mismatch sideband lands on a bin. The
    /// pair-sum holds 0.81-1 of a tone's power at any grid phase.
    double peak_interp;
};

/// Dimensions of an accumulated acquisition surface (see @ref channelized_accumulate).
/// The surface is a flat [n_dop][Mp][fine()] array of |D|^2, indexed
/// `surf[(d*Mp + q)*fine() + i]`.
///
/// @c s_stored is usually @c sph, but smaller when the covering channels' frequency
/// differences share a factor with @c sph: the fine-lag axis is then exactly periodic and the
/// extra columns would be copies (see @ref aggregate_accumulate). The fine lag still strides
/// by @c sph in absolute delay; only the stored width shrinks, so index with fine() and do
/// delay arithmetic with @c sph.
///
/// @c s_step decimates the stored width. The fine-lag response is the beam of the covering
/// comb, whose main lobe reaches its first null sph/(comb span in bins) samples from the peak
/// (157 samples for a 104-bin span at sph 16384), so storing every sample oversamples it. A
/// step of 32 costs at most ~0.15 dB of scalloping, and the refine that follows the acquire
/// rescans +-refine_span samples anyway.
struct AcquisitionSurface {
    int n_dop;        ///< Doppler trials (= doppler_grid.size())
    int Mp;           ///< replica hop period (coarse-lag range)
    int sph;          ///< full-rate samples per hop (the fine-lag stride in absolute delay)
    int s_stored = 0; ///< fine-lag period in samples (sph when not periodic; 0 is read as sph)
    int s_step = 1;   ///< samples per stored column (1 = every sample)
    /// Stored fine-lag columns, ceil(s_stored / s_step).
    int fine() const {
        const int w = s_stored > 0 ? s_stored : sph;
        const int st = s_step > 0 ? s_step : 1;
        return (w + st - 1) / st;
    }
    long size() const {
        return (long)n_dop * Mp * fine();
    }
};

/// Direction of the fine-lag axis relative to the coarse (hop) axis: -1, opposite. It is a
/// property of the analysis bank, not of the acquire: a bank that folds newest-first
/// (dsp::pfb_push) into its DFT, as ChannelizedReplicaBank does when it models the CHORD
/// F-engine and builds the replica the acquire correlates against, runs its fine lag against
/// the coarse lag. A bank that folds oldest-first would need +1.
///
/// A wrong sign offsets the coarse code phase by up to one hop (52.4 chips of a 10.23 Mchip/s
/// code at CHORD), which only a refine spanning a full hop absorbs.
constexpr int FINE_LAG_SIGN_PFB = -1;

/**
 * Add one data window's |D|^2 surface into @c surf (incoherent integration) and return the
 * surface dimensions (PRN-independent) for @ref channelized_peak. Pass the same @c surf across
 * consecutive windows to integrate; the peak bin is stationary when each window spans a whole
 * number of code periods (true for a window of @c Mp hops), so the weak-signal surface SNR grows
 * as sqrt(windows). @c surf is resized and zeroed when its size does not match. The Doppler phase
 * reference restarts on each call, so only |D|^2 is summed. Runs @ref channel_correlate on each
 * covering channel, then @ref aggregate_accumulate.
 *
 * @param data_ch    Channelized data, [N][M] (N channels, M hops).
 * @param repl0_ch   Code-only replica (code phase 0, Doppler 0), analyzed through
 *                   the identical bank, [N][Mp]. Its hop length Mp is the replica
 *                   period and sets the coarse-lag search range; it may exceed the
 *                   data window M (the code is Mp-periodic in the hop index).
 * @param covering   Channel indices carrying the carrier (others are noise-only
 *                   and excluded from the coherent sum).
 * @param doppler_grid Trial Doppler frequencies, Hz.
 * @param sample_rate  Full (pre-channelization) sample rate, Hz.
 * @param num_chan     N, the channel count.
 * @param surf        The |D|^2 surface to add into, [n_dop][Mp][fine()].
 * @param ws          Caller-owned FFTW workspace, reused across calls.
 * @param chan_freq   Optional integer frequency index of each covering channel,
 *                    used in the cross-channel fine-lag ramp. Empty means @c covering
 *                    (channel c at frequency index c, the r2c natural order). The
 *                    per-channel analysis phase cancels in P_c, so only this index matters.
 * @param samples_per_hop  Full-rate samples consumed per hop (the decimation).
 *                    0 means num_chan, a critically sampled complex bank where the
 *                    hop equals the channel count. An r2c real-FFT bank consumes
 *                    2N real samples per hop, so pass 2*num_chan there.
 * @param n_threads   Threads for both halves.
 * @param fine_step   Fine-lag decimation (see AcquisitionSurface::s_step).
 */
AcquisitionSurface
channelized_accumulate(const std::vector<std::vector<std::complex<float>>>& data_ch,
                       const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                       const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                       double sample_rate, int num_chan, std::vector<double>& surf,
                       AcquireWorkspace& ws, const std::vector<int>& chan_freq = {},
                       int samples_per_hop = 0, int n_threads = 1, int fine_step = 1);

/// Coarse correlation for one channel, the per-channel half of @ref channelized_accumulate:
/// P[d][q] = IFFT{ FFT(wiped_d(data)) * conj(FFT(repl0)) } / Mp for each Doppler d, with the
/// M data hops zero-padded to the Mp-hop replica. The cross-channel combine is
/// @ref aggregate_accumulate. @c samples_per_hop is the decimation (2N for an r2c bank) used
/// in the Doppler wipe.
std::vector<std::vector<std::complex<float>>>
channel_correlate(const std::vector<std::complex<float>>& data,
                  const std::vector<std::complex<float>>& repl0,
                  const std::vector<double>& doppler_grid, double sample_rate, int samples_per_hop,
                  AcquireWorkspace& ws);

/// As @ref channel_correlate, but into a caller-owned @c P that is resized only when its shape
/// changes, so a caller looping over windows allocates once rather than per iteration.
void channel_correlate_into(const std::vector<std::complex<float>>& data,
                            const std::vector<std::complex<float>>& repl0,
                            const std::vector<double>& doppler_grid, double sample_rate,
                            int samples_per_hop, AcquireWorkspace& ws,
                            std::vector<std::vector<std::complex<float>>>& P);

/// Cross-channel fine-lag DFT and incoherent |D|^2 accumulation, the cross-channel half of
/// @ref channelized_accumulate. Given per-channel correlations @c P [n_covering][n_dop][Mp]
/// and the covering channels' frequency indices @c chan_freq, adds the |D(q,s)|^2 surface into
/// @c surf (pass the same surf across snapshots to integrate) and returns its dimensions.
/// @c fine_step decimates the fine-lag axis (see AcquisitionSurface::s_step).
AcquisitionSurface
aggregate_accumulate(const std::vector<std::vector<std::vector<std::complex<float>>>>& P,
                     const std::vector<int>& chan_freq, int samples_per_hop,
                     std::vector<double>& surf, int n_threads = 1, int fine_step = 1);

/// Surface dimensions for a covering set, without computing a surface. @ref aggregate_accumulate
/// takes its layout (the s_stored rule) from here, so a caller that never builds a host surface
/// (the GPU search) indexes its reduction the same way.
AcquisitionSurface surface_dims(const std::vector<int>& chan_freq, int samples_per_hop, int n_dop,
                                int Mp, int fine_step);


/// Reduce an accumulated surface to its peak: code phase, Doppler, lag and SNR (peak / surface
/// mean), via @ref peak_from_reduction.
/// @param pairsum_select  choose the Doppler plane by the mainlobe pair-sum (local maxima only)
///                        rather than the raw argmax; see the implementation.
/// @param data_hops       data hops per window when fewer than Mp (0 = Mp): sets the
///                        zero-padded response that the sub-grid Doppler refine inverts.
AcquisitionResult channelized_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length, bool pairsum_select = false,
                                   int data_hops = 0);

/// The second half of @ref channelized_peak: everything after the surface scan. A caller that
/// reduces the surface some other way (the GPU search never materializes it on the host) reuses
/// this rather than reimplementing it, because the cell -> code-phase mapping has two sign
/// inversions (the fine half runs opposite to the coarse half; the reported delay is the
/// negative of the surface lag) that a second copy could get wrong.
///
/// @param peak     value of the chosen cell: the surface max, or the chosen plane's max when
///                 the plane was picked by pair-sum
/// @param mean     mean cell value over the whole surface (the SNR denominator)
/// @param best_d,best_q,best_i  the chosen cell, decomposed
/// @param dop_peak per-Doppler max over the lag, length n_dop: the input of the sub-grid
///                 Doppler refine and of peak_interp. Empty skips both (the Doppler stays on
///                 the grid and peak_interp = peak).
/// @param dop_loc_m,dop_loc_p  the neighbour Doppler planes (best_d -+ 1) sampled at the
///                 winning lag (max over a few cells around (best_q, best_i)), not their
///                 plane-wide max. When both are >= 0 and below the peak, the refine inverts
///                 the amplitude ratio of the zero-padded response sin(pi x u)/x on these;
///                 otherwise (-1 is the default) it fits a parabola to dop_peak if concave.
/// @param dop_u    data window as a fraction of the Mp-hop transform (data_hops/Mp), the u of
///                 that response.
AcquisitionResult peak_from_reduction(const AcquisitionSurface& dims,
                                      const std::vector<double>& doppler_grid, double sample_rate,
                                      double chip_rate, long code_length, double peak, double mean,
                                      int best_d, int best_q, int best_i,
                                      const std::vector<double>& dop_peak, double dop_loc_m = -1.0,
                                      double dop_loc_p = -1.0, double dop_u = 1.0);

} // namespace gnss

#endif // GNSS_CHANNELIZED_ACQUIRE_HPP
