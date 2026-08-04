/**
 * @file
 * @brief Shared channelized GNSS replica generator: PFB-analyze a PRN's replica
 *        through the F-engine's exact bank, in the channelized domain.
 *  - gnss::ChannelizedReplicaBank
 */

#ifndef GNSS_CHANNELIZED_REPLICA_HPP
#define GNSS_CHANNELIZED_REPLICA_HPP

#include "gnssSignal.hpp"   // for SignalDescriptor
#include "pfbPrototype.hpp" // for Window

#include <complex>    // for complex
#include <cstdint>    // for int8_t
#include <fftw3.h>    // for fftwf_plan, fftwf_complex
#include <functional> // for function
#include <vector>     // for vector

namespace gnss {

/**
 * @class ChannelizedReplicaBank
 * @brief Generates a GNSS replica and PFB-analyzes it through the same r2c bank
 *        the F-engine uses, returning per-channel hop streams.
 *
 * The measurement and search tiers of the distributed-band pipeline both need the
 * replica in the channelized domain: a real passband replica @c code*cos(carrier)
 * pushed through the identical @ref fftwEngine PFB (real-input r2c, @c 2N real
 * samples/hop, @c spectrum_length positive-frequency channels in natural order).
 * The replica is anchored to the *absolute* sample index, so it composes with the
 * channelized data exactly (same inter-channel phases). Owns its FFTW plan/buffers
 * and the cached spreading codes; not thread-safe (one bank per worker thread).
 *
 * @author Keith Vanderlinde
 */
class ChannelizedReplicaBank {
public:
    /// @param sig            signal descriptor (chip rate, code length/period).
    /// @param sample_rate    full (pre-channelization) sample rate, Hz.
    /// @param f_offset       carrier offset from band centre, Hz.
    /// @param spectrum_length channels per hop (the F-engine N).
    /// @param num_taps       PFB taps (must match the F-engine).
    /// @param window         PFB prototype window (must match the F-engine).
    /// @param prns           PRN list; @c channels()/@c code_chip() index into it.
    ChannelizedReplicaBank(const SignalDescriptor& sig, double sample_rate, double f_offset,
                           int spectrum_length, int num_taps, dsp::Window window,
                           const std::vector<int>& prns);
    ~ChannelizedReplicaBank();
    ChannelizedReplicaBank(const ChannelizedReplicaBank&) = delete;
    ChannelizedReplicaBank& operator=(const ChannelizedReplicaBank&) = delete;

    /// Replica for PRN index @c p over @c n_hops hops from @c window_start_sample at
    /// the given code phase + Doppler; returns [spectrum_length][n_hops] channels.
    /// @c nh_phase aligns the secondary (Neuman-Hofman) overlay: the +-1 overlay chip for
    /// absolute primary-period index @c k is @c secondary[(k + nh_phase) mod len]. NEGATIVE
    /// (the default) leaves the overlay OFF -> the RAW despread (so a downstream search, e.g.
    /// @ref overlay_wipe, can find the alignment from the per-record A); set it >=0 once the
    /// alignment is known (from GPS time, or that search). No effect for signals without a
    /// secondary code (L1 C/A, L2C).
    std::vector<std::vector<std::complex<float>>>
    channels(int p, long long window_start_sample, double code_phase_chips, double doppler_hz,
             int n_hops, int nh_phase = -1);

    /// Hop-rate channelized replica for the listed channels -- numerically EQUAL to
    /// @ref channels() (to ~machine precision) but built per chip, not per sample. The
    /// spreading code is constant over a chip (~Fs/chip_rate samples), so the polyphase
    /// filter collapses to a per-chip sum: a chip's filter contribution is the cumulative
    /// filter @c Phi[k_hi]-Phi[k_lo-1] over its INTEGER tap range (the chip boundaries land
    /// between samples, so this is exact -- no interpolation). Carrier = a per-hop output
    /// phasor + the channel-carrier offset baked into the (slowly-rebuilt) filter; both
    /// images kept. Each hop is O(n_chips) MACs vs O(num_taps*fft_len) per sample; the
    /// O(num_taps*fft_len) prefix-sum amortizes over a stream.
    ///
    /// @c nav_bit (optional): the +-1 data bit for an absolute chip index. The 20 ms nav
    /// edge falls on a code-period (=chip) boundary, so multiplying it in PER CHIP gives the
    /// EXACT data-wiped replica @c d*R_c (a per-hop post-multiply would smear the ~num_taps
    /// hops straddling an edge by ~100%). Returns [want.size()][n_hops].
    std::vector<std::vector<std::complex<float>>>
    channels_hoprate(int p, long long window_start_sample, double code_phase_chips,
                     double doppler_hz, int n_hops, const std::vector<int>& want,
                     const std::function<float(long long)>& nav_bit = {}, int nh_phase = -1) const;

    /// The slowly-varying half of the hop-rate generator: the cumulative channel filters
    /// (both carrier images) for @c want at @c doppler_hz. Depends only on the carrier
    /// offset, so it barely moves with Doppler -- rebuild only every ~tens of Hz of drift.
    /// @ref HopRateReplicaStream caches it; @ref hoprate_stream consumes it.
    struct HopRateFilter {
        double doppler_hz = 0.0;
        int prn_index = -1; ///< PRN this filter's carrier was built for (FDMA); -1 = band-wide
        int n_chips = 0;
        std::vector<int> chans;
        std::vector<std::vector<std::complex<double>>> PhiA, PhiB; ///< [channel][Lf+1]
    };
    /// @c prn_index selects that PRN's carrier offset (FDMA); -1 = the band offset alone.
    /// ⚠️ Under FDMA the filter is PER-PRN (different satellites sit on different carriers), so
    /// a filter built for one PRN must not be streamed for another -- @ref hoprate_stream
    /// checks. @ref HopRateReplicaStream already owns one filter per PRN, so steady-state cost
    /// is unchanged; only a shared-filter caller has to start passing an index.
    HopRateFilter hoprate_filter(const std::vector<int>& want, double doppler_hz,
                                 int prn_index = -1) const;

    /// Stream @c n_hops from a prebuilt @ref HopRateFilter. The per-hop carrier phasor +
    /// code phase use the CURRENT @c code_phase_chips / @c doppler_hz (exact); the filter
    /// shape comes from @c f (built for a nearby Doppler). @c nav_bit per chip = exact wipe.
    std::vector<std::vector<std::complex<float>>>
    hoprate_stream(const HopRateFilter& f, int p, long long window_start_sample,
                   double code_phase_chips, double doppler_hz, int n_hops,
                   const std::function<float(long long)>& nav_bit = {}, int nh_phase = -1) const;

    /// Global channel indices whose passband covers the carrier at @c doppler_hz.
    /// @c prn_index selects that PRN's carrier offset (FDMA); -1 = the band offset alone.
    std::vector<int> covering_bins(double doppler_hz, double doppler_margin_hz,
                                   int prn_index = -1) const;

    // ---- FDMA support (GLONASS L1OF/L2OF) --------------------------------------------
    // ★ Every other constellation we carry is CDMA: one carrier for the whole band, and the
    // per-satellite identity lives in the code. GLONASS FDMA inverts that -- every satellite
    // transmits the SAME 511-chip code and is separated by CARRIER, satellite k sitting at
    // 1246.0 + k*0.4375 MHz on L2. The whole comb (k = -7..+6, 5.69 MHz) fits inside one
    // 10 MHz tune, so this needs no retuning -- but the bank's single @c _f_offset has to
    // become per-PRN, which is what these two do.
    //
    // Everything below defaults to the previous behaviour: an unset PRN contributes 0 Hz, and
    // the new @c prn_index parameters default to -1, so every CDMA caller is untouched.

    /// Extra carrier offset for PRN index @c p, Hz, on top of @c f_offset (0 = CDMA default).
    void set_prn_freq_offset(int prn_index, double df_hz);

    /// Total carrier offset from band centre for PRN index @c p: @c f_offset + its FDMA
    /// offset. @c p < 0 (or unset) gives the band offset alone.
    double carrier_offset(int prn_index) const;

    /// Bipolar code chip for PRN index @c p at fractional chip phase (wraps period).
    int8_t code_chip(int p, double chip_phase) const;

    /// Secondary (Neuman-Hofman) overlay length in primary periods (NH10=10 on L5 I5,
    /// NH20=20 on L5 Q5); 0 if the signal has no overlay. Also the number of @c nh_phase
    /// alignments to search.
    int secondary_length() const { return _secondary_length; }

    /// Bipolar (+-1) secondary-overlay chip for absolute primary-period index @c period at
    /// alignment @c nh_phase; +1 if the signal has no secondary code (so it composes as a no-op).
    int overlay_sign(long long period, int nh_phase) const;

    /// Code-Doppler feed-forward sign (+1 nominal: approaching -> faster code, as the
    /// carrier). Flip to -1 if the r2c fold inverts it vs the seed convention; set by
    /// the stage from config `code_doppler_sign` so it can be tuned without a rebuild.
    double code_doppler_sign = 1.0;

    int spectrum_length() const { return _N; }
    int fft_len() const { return _fft_len; }
    double f_offset() const { return _f_offset; }
    int repl_period_hops() const { return _repl_period_hops; }
    double chip_rate_hz() const { return _sig.chip_rate_hz; }
    double carrier_hz() const { return _sig.carrier_hz; } ///< sky carrier (for code-Doppler)
    long code_length() const { return _sig.code_length; }
    // Combined-stream (TDM zero-stuffed) quantities + the raw code table -- what an external
    // (GPU) despread needs to reproduce hoprate_stream exactly (see GnssCudaDespread).
    double eff_chip_rate() const { return _eff_chip_rate; }
    long eff_code_length() const { return _eff_code_length; }
    int comb_mult() const { return _comb_mult; }
    const std::vector<int8_t>& full_code(int p) const { return _full_code[(size_t)p]; }

private:
    SignalDescriptor _sig;
    double _sample_rate;
    std::vector<double> _prn_df; ///< per-PRN FDMA carrier offset, Hz (empty/0 = CDMA)
    double _f_offset;
    int _N;
    int _fft_len;   ///< 2*N real samples per hop (r2c)
    int _num_taps;
    int _repl_period_hops; ///< code_samples / gcd(fft_len, code_samples)
    // Time-multiplexed signals (L2C CM/CL) interleave their component with a sibling at
    // 2x the component chip rate. We model the COMBINED stream: the component code placed
    // at its tdm_phase parity of the combined chips, zeros at the sibling's. So the bank
    // works at the combined rate/length, and _full_code holds the combined (zero-stuffed)
    // sequence. comb_mult=1 (eff_*=_sig.*) for ordinary signals -> exact no-op.
    int _comb_mult;          ///< 2 if time_multiplexed, else 1
    double _eff_chip_rate;   ///< _sig.chip_rate_hz * _comb_mult (combined chipping rate)
    long _eff_code_length;   ///< _sig.code_length * _comb_mult (combined-stream length)
    std::vector<float> _proto;
    std::vector<std::vector<int8_t>> _full_code;
    std::vector<int8_t> _secondary;   ///< +-1 Neuman-Hofman overlay (NH10/NH20); empty if none
    int _secondary_length = 0;        ///< overlay period in primary periods; 0 = no overlay

    float* _fold;
    fftwf_complex* _spec;
    fftwf_plan _p_fwd;
    std::vector<float> _replica_hist;
};

/**
 * @class HopRateReplicaStream
 * @brief Realtime streaming wrapper around @ref ChannelizedReplicaBank::hoprate_stream.
 *
 * Holds the cumulative channel filter (the O(num_taps*fft_len) part) and reuses it across
 * @ref generate() calls, rebuilding it only when the carrier Doppler has drifted past
 * @c refresh_hz -- so steady-state generation is O(n_chips) MACs/hop. One stream per PRN
 * per channel set; not thread-safe. The bank it references must outlive it.
 *
 * @author Keith Vanderlinde
 */
class HopRateReplicaStream {
public:
    /// @param bank        the replica bank (must outlive this; provides codes/filter).
    /// @param prn_index   PRN index into the bank.
    /// @param channels    global channel indices to generate (the covering set).
    /// @param refresh_hz  rebuild when |doppler - built| exceeds this -- it bounds the filter
    ///                    staleness (~-54 dB at 20 Hz, ~linear), so it sets the peel depth.
    ///                    The rebuild is cheap and amortizes over ~seconds (Doppler drifts
    ///                    ~1 Hz/s), so run tight: a few Hz for a deep peel of a +60 dB source.
    HopRateReplicaStream(const ChannelizedReplicaBank& bank, int prn_index,
                         std::vector<int> channels, double refresh_hz = 2.0);

    /// Next @c n_hops from @c window_start at code phase @c cp and carrier @c doppler_hz.
    /// Rebuilds the cached filter on a large Doppler step; otherwise reuses it. @c nav_bit
    /// applies the +-1 data per chip (exact wipe). Returns [channels][n_hops] (owned here).
    const std::vector<std::vector<std::complex<float>>>&
    generate(long long window_start, double cp, double doppler_hz, int n_hops,
             const std::function<float(long long)>& nav_bit = {});

    /// Filter rebuilds so far (diagnostics; ~0 in steady lock).
    long rebuilds() const { return _rebuilds; }

private:
    const ChannelizedReplicaBank& _bank;
    int _prn;
    std::vector<int> _chans;
    double _refresh_hz;
    bool _built = false;
    long _rebuilds = 0;
    ChannelizedReplicaBank::HopRateFilter _filter;
    std::vector<std::vector<std::complex<float>>> _out;
};

} // namespace gnss

#endif // GNSS_CHANNELIZED_REPLICA_HPP
