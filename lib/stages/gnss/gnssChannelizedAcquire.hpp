/**
 * @file
 * @brief Channelized acquisition: find a GNSS signal's code phase + Doppler
 *        from PFB channels alone, with no wideband resynthesis.
 *
 * The measurement tier (@ref gnssChannelizedDespread.hpp) needs a seed code
 * phase + Doppler. This produces them from the same channelized voltages a node
 * already holds. The key is recovering *full* code-phase resolution: each
 * channel is decimated by @c sph (the full-rate samples per hop), so a per-channel
 * correlation resolves the lag only to a hop -- far coarser than one chip. Writing
 * the lag as @f$ \tau = q\,\mathrm{sph} + s @f$, the sub-hop part @c s appears as a
 * phase ramp across channels, so
 * @f[ D(q\,\mathrm{sph}+s) = \sum_c P_c(q)\,e^{+i 2\pi c s / \mathrm{sph}}, \qquad
 *     P_c(q) = \sum_m X_c[m]\,\overline{R_c[m-q]} . @f]
 * The coarse lag @c q comes from a per-channel circular correlation; the fine
 * lag @c s from an sph-point DFT of @c {P_c(q)} across channels. The circular
 * correlation wraps the replica at its true hop-period @c Mp (= the length of
 * @c repl0_ch), which lets the data window @c M and the code periodicity differ
 * -- the fractional-hop-per-period case (e.g. r2c N=40: 62.5 hops/period). That cross-channel
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
#include <fftw3.h> // for fftwf_plan, fftwf_complex
#include <vector>  // for vector

namespace gnss {

/**
 * Reusable FFTW workspace for the FFT-based coarse correlation in
 * @ref channelized_accumulate. Holds length-@c Mp complex plans + scratch
 * buffers; create one and reuse it across windows (it self-sizes on first use,
 * and re-plans only if @c Mp changes). Non-copyable; owns its FFTW resources.
 * Plan create/destroy take the shared FFTW planner lock; execution is lock-free.
 */
struct AcquireWorkspace {
    AcquireWorkspace() = default;
    ~AcquireWorkspace();
    AcquireWorkspace(const AcquireWorkspace&) = delete;
    AcquireWorkspace& operator=(const AcquireWorkspace&) = delete;
    /// (Re)allocate plans + buffers for length-@c n circular transforms.
    void ensure(int n);

    int Mp = 0;                  ///< current transform length (0 = unallocated)
    fftwf_complex* in = nullptr; ///< scratch input  (length Mp)
    fftwf_complex* out = nullptr;///< scratch output (length Mp)
    fftwf_plan fwd = nullptr;    ///< length-Mp c2c forward  (in -> out)
    fftwf_plan inv = nullptr;    ///< length-Mp c2c backward (in -> out, unnormalized)
};

/// Outcome of a channelized acquisition search.
struct AcquisitionResult {
    double code_phase_chips; ///< peak code phase, chips
    double doppler_hz;       ///< peak Doppler from the grid, Hz
    long peak_tau_samples;   ///< peak lag, full-rate samples (= q*N + s)
    double peak;             ///< peak |D|^2
    double snr;              ///< peak / mean of the |D|^2 surface
};

/// Dimensions of an accumulated acquisition surface (see @ref channelized_accumulate).
/// The surface is a flat [n_dop][Mp][s_stored] array of |D|^2, indexed
/// `surf[(d*Mp + q)*s_stored + s]`.
///
/// @c s_stored is usually @c sph, but is SMALLER when the covering channel indices share a
/// common factor with @c sph, because the fine-lag axis is then exactly periodic and the extra
/// columns are bit-for-bit copies -- see @ref channelized_accumulate. The fine lag still
/// STRIDES by @c sph when converting a surface cell to an absolute delay (tau = q*sph + s);
/// only the stored width shrinks. Use @c s_stored for indexing and @c sph for delay arithmetic.
struct AcquisitionSurface {
    int n_dop;        ///< Doppler trials (= doppler_grid.size())
    int Mp;           ///< replica hop-period (coarse-lag range)
    int sph;          ///< full-rate samples per hop (the fine-lag STRIDE in absolute delay)
    int s_stored = 0; ///< fine-lag columns actually stored (== sph when not periodic)
    /// Distinct fine-lag columns, tolerating a surface built before @c s_stored existed.
    int fine() const { return s_stored > 0 ? s_stored : sph; }
    long size() const { return (long)n_dop * Mp * fine(); }
};

/**
 * Search the (code phase, Doppler) surface from channelized voltages.
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
 * @param chip_rate    Spreading code chip rate, Hz.
 * @param num_chan     N, the channel count.
 * @param code_length  Chips per code period (for wrapping the code phase).
 * @param chan_freq   Optional integer frequency index of each covering channel,
 *                    used in the cross-channel fine-lag ramp. Empty defaults to
 *                    `covering` (channel c <-> +frequency c, the clean bank --
 *                    which is also the r2c natural-order bank). The per-channel
 *                    analysis phase cancels in P_c, so only this index matters.
 * @param samples_per_hop  Full-rate samples consumed per hop (the decimation).
 *                    0 -> num_chan, a critically-sampled complex bank where the
 *                    hop equals the channel count. An r2c real-FFT bank consumes
 *                    2N real samples per hop, so pass 2*num_chan there.
 */
AcquisitionResult
channelized_acquire(const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                    const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                    double sample_rate, double chip_rate, int num_chan, long code_length,
                    const std::vector<int>& chan_freq = {}, int samples_per_hop = 0);

/**
 * Add one data window's |D|^2 acquisition surface into @c surf (incoherent
 * integration). Pass the SAME @c surf across consecutive windows to integrate;
 * the peak bin is stationary when each window spans an integer number of code
 * periods (true for a window of @c Mp hops), so the weak-signal surface SNR
 * grows as sqrt(windows). @c surf is (re)sized + zeroed on first use.
 *
 * Same conventions and arguments as @ref channelized_acquire; returns the
 * surface dimensions (PRN-independent) for @ref channelized_peak. The per-window
 * Doppler phase reference resets each call, so only |D|^2 (magnitude) is summed.
 *
 * The coarse-lag correlation is done in the Fourier domain
 * (@c P = IFFT{FFT(data) * conj(FFT(replica))}); @c ws is a caller-owned FFTW
 * workspace reused across calls (the replica FFT is recomputed per window since
 * the replica advances, but the plans/buffers persist).
 */
AcquisitionSurface
channelized_accumulate(const std::vector<std::vector<std::complex<float>>>& data_ch,
                       const std::vector<std::vector<std::complex<float>>>& repl0_ch,
                       const std::vector<int>& covering, const std::vector<double>& doppler_grid,
                       double sample_rate, int num_chan, std::vector<double>& surf,
                       AcquireWorkspace& ws, const std::vector<int>& chan_freq = {},
                       int samples_per_hop = 0, int n_threads = 1);

/// Per-channel coarse correlation for ONE channel -- the distributable half of the
/// search. P_c[d][q] = IFFT{ FFT(wiped0_d) * conj(FFT(repl0)) } for each Doppler d
/// (data = M hops, repl0 = Mp hops). The cross-channel combine is done separately by
/// @ref aggregate_accumulate, so this can run per channel on a different node.
/// @c samples_per_hop is the decimation (2N for an r2c bank) used in the Doppler wipe.
std::vector<std::vector<std::complex<float>>>
channel_correlate(const std::vector<std::complex<float>>& data,
                  const std::vector<std::complex<float>>& repl0,
                  const std::vector<double>& doppler_grid, double sample_rate, int samples_per_hop,
                  AcquireWorkspace& ws);

/// Cross-channel fine-lag DFT + incoherent |D|^2 accumulation -- the central
/// aggregation half. Given per-channel correlations @c P [n_covering][n_dop][Mp]
/// and the covering channels' global frequency indices @c chan_freq, add the
/// |D(q,s)|^2 surface into @c surf (pass the same surf across snapshots to integrate).
/// Returns the surface dimensions.
AcquisitionSurface
aggregate_accumulate(const std::vector<std::vector<std::vector<std::complex<float>>>>& P,
                     const std::vector<int>& chan_freq, int samples_per_hop,
                     std::vector<double>& surf, int n_threads = 1);

/// Reduce an accumulated surface to its peak: code phase, Doppler, lag and SNR
/// (peak / surface mean). Mirrors @ref channelized_acquire's peak bookkeeping.
AcquisitionResult channelized_peak(const std::vector<double>& surf,
                                   const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length);

} // namespace gnss

#endif // GNSS_CHANNELIZED_ACQUIRE_HPP
