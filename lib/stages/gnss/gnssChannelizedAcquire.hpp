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

#include "gnssChannelizedReplica.hpp" // for ChannelizedReplicaBank (ms_split_accumulate)

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
/// @c s_step DECIMATES that stored width. The fine-lag response is the beam of the covering
/// comb, so its main lobe is @c sph/(comb span in bins) wide -- 157 samples for CHORD's 27
/// channels over 104 bins -- while the axis is stored at 1-sample resolution. Sampling a
/// 157-sample lobe every sample is ~157x oversampled, and the surface is the whole cost of a
/// pass (2.8 GB and 15 Tflop per pass at CHORD scale, both of which fit the ~880 s observed).
/// Stepping s costs only scalloping: at step 32 the worst-case peak offset is 16 samples,
/// ~0.15 dB. Nothing downstream needs better, because the refine that follows the acquire
/// rescans +-refine_span samples anyway. This is NOT the coarse-axis fold that was tried and
/// reverted (see docs/CHORD_GNSS_STATE.md 5o): it assumes no periodicity, so the secondary
/// code cannot invalidate it.
struct AcquisitionSurface {
    int n_dop;        ///< Doppler trials (= doppler_grid.size())
    int Mp;           ///< replica hop-period (coarse-lag range)
    int sph;          ///< full-rate samples per hop (the fine-lag STRIDE in absolute delay)
    int s_stored = 0; ///< fine-lag WIDTH in samples (== sph when not periodic)
    int s_step = 1;   ///< samples per stored column (1 = every sample, the old behaviour)
    /// Distinct fine-lag columns, tolerating a surface built before @c s_stored existed.
    int fine() const {
        const int w = s_stored > 0 ? s_stored : sph;
        const int st = s_step > 0 ? s_step : 1;
        return (w + st - 1) / st;
    }
    /// Absolute delay of surface cell (q, i), full-rate samples.
    long tau(int q, int i) const { return (long)q * sph + (long)i * (s_step > 0 ? s_step : 1); }
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
                       int samples_per_hop = 0, int n_threads = 1, int fine_step = 1);

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
/// @c fine_step decimates the fine-lag axis (1 = every sample; see AcquisitionSurface::s_step).
AcquisitionSurface
aggregate_accumulate(const std::vector<std::vector<std::vector<std::complex<float>>>>& P,
                     const std::vector<int>& chan_freq, int samples_per_hop,
                     std::vector<double>& surf, int n_threads = 1, int fine_step = 1);

/// Reduce an accumulated surface to its peak: code phase, Doppler, lag and SNR
/// (peak / surface mean). Mirrors @ref channelized_acquire's peak bookkeeping.
/// SUB-WINDOW ("ms-split") ACCUMULATION -- docs/CHORD_GNSS_MS_SPLIT_SEARCH.md.
///
/// The ordinary acquire integrates coherently over ONE REPLICA PERIOD, because that is what
/// makes its cyclic correlation exact. On airspy that period IS one code period (fft_len
/// divides code_samples: 250 hops = 1 ms). On CHORD it is SIXTEEN of them (195.3125 hops per
/// code period), and that single fact inflates three axes at once: the coarse-lag range (3125
/// vs ~196), the Doppler grid (1/T resolution: 62.5 Hz vs 1000 Hz) and the NH alignment search
/// (a 16 ms window straddles 16 of the 20 overlay chips at an unknown offset; a 1 ms window
/// spans exactly one -- a constant sign, invisible to |D|^2). Nobody chose a 16 ms window;
/// Mp chose it.
///
/// This integrates ~1 ms sub-windows instead and sums |D|^2 across them. Three things make it
/// work, and each is where a naive version goes wrong:
///
///  1. FULL OVERLAP AT EVERY LAG. The replica spans TWO code periods and the data ONE, so every
///     lag in [0, one period) sees all N data hops. Zero-padding only the replica gives a
///     triangular weighting -- SNR falling off across the lag axis, which reads as a weak
///     satellite rather than as a bug.
///  2. NO CYCLIC WRAP. A 196-hop replica is NOT periodic over 196 hops (one code period is
///     195.3125), so a cyclic correlation at that length wraps 36.0 chips wrong.
///  3. NO PER-SUB-WINDOW ROLL. Each sub-window's replica is generated at ITS OWN absolute start
///     with argument 0, and every generator here forms C(n) = arg + n*cps over the ABSOLUTE
///     sample index -- so the peak lands at the same lag (the satellite's argument) in every
///     sub-window, with nothing to realign. Same property that makes seed transport work
///     (ChannelizedReplicaBank::phase_from_arg), used a second time.
///
/// @param sub_hops   N, hops per sub-window: ceil(code period in hops) = 196 at CHORD.
/// @param n_sub      K, how many sub-windows to accumulate (100 = ~100 ms).
/// @return surface dimensions; @c surf accumulates |D|^2 exactly as channelized_accumulate does.
AcquisitionSurface
ms_split_accumulate(gnss::ChannelizedReplicaBank& bank, int prn_index,
                    const std::vector<std::vector<std::complex<float>>>& data_ch,
                    const std::vector<int>& chan_ids, long long window_start_sample, int sub_hops,
                    int n_sub, const std::vector<double>& doppler_grid, double sample_rate,
                    std::vector<double>& surf, AcquireWorkspace& ws, int fine_step = 1,
                    int n_threads = 1);

AcquisitionResult channelized_peak(const std::vector<double>& surf,
                                   const AcquisitionSurface& dims,
                                   const std::vector<double>& doppler_grid, double sample_rate,
                                   double chip_rate, long code_length);

/// Peak-pick an ms-split surface. NOT channelized_peak: that one's tau -> code-phase mapping
/// silently assumes two things the shipped geometry arranges and the ms-split cannot.
///
///  1. THE REPLICA STARTS AT CODE PHASE ~0. The search anchors repl0 at Mp*fft_len = 51,200,000
///     samples = 16 exact code periods, so its phase there is 52.4 chips (one hop) and the
///     following refine absorbs it. An ms-split replica has to sit N hops before ITS data
///     window, at an arbitrary absolute sample ~1.9e15, whose phase is an arbitrary number --
///     7611 chips at the bench point. Left uncorrected the reported phase is wrong by that
///     amount, and it MOVES with the snapshot: -52.3776 chips per hop of uptime.
///  2. Mp*sph IS A WHOLE NUMBER OF CODE PERIODS. True for the shipped 3125*16384 (16 periods),
///     false for 2N*16384 (2.007 periods at N=196). At the peak the cyclic correlation reads
///     replica indices (m-q) mod Mp, i.e. a whole Mp hops further on, so that 0.007 of a period
///     -- 72.0 chips -- lands directly in the answer.
///
/// It also folds the fine index signed and restricts the coarse lag to the full-overlap window;
/// see the implementation for why each is necessary, and what measured it.
///
/// @param phi_r0   the replica's own code phase at its index 0, for sub-window 0:
///                 @c bank.phase_from_arg(0.0, window_start_sample - sub_hops*fft_len, 0.0).
/// @param sub_hops N, the same value handed to @ref ms_split_accumulate.
AcquisitionResult ms_split_peak(const std::vector<double>& surf, const AcquisitionSurface& dims,
                                const std::vector<double>& doppler_grid, double sample_rate,
                                double chip_rate, long code_length, double phi_r0, int sub_hops);

} // namespace gnss

#endif // GNSS_CHANNELIZED_ACQUIRE_HPP
