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
        int n_chips = 0;
        std::vector<int> chans;
        std::vector<std::vector<std::complex<double>>> PhiA, PhiB; ///< [channel][Lf+1]
    };
    HopRateFilter hoprate_filter(const std::vector<int>& want, double doppler_hz) const;

    /// Stream @c n_hops from a prebuilt @ref HopRateFilter. The per-hop carrier phasor +
    /// code phase use the CURRENT @c code_phase_chips / @c doppler_hz (exact); the filter
    /// shape comes from @c f (built for a nearby Doppler). @c nav_bit per chip = exact wipe.
    std::vector<std::vector<std::complex<float>>>
    hoprate_stream(const HopRateFilter& f, int p, long long window_start_sample,
                   double code_phase_chips, double doppler_hz, int n_hops,
                   const std::function<float(long long)>& nav_bit = {}, int nh_phase = -1) const;

    /// @ref hoprate_stream writing into a CALLER-OWNED buffer, resized only when it does not
    /// already fit. The refine evaluates ~14 trial phases in an OpenMP loop and the by-value
    /// form allocates ~2.6 MB inside every iteration, so 14 threads hit the allocator at once,
    /// every detection. That exact pathology was diagnosed in the sibling ms-split path
    /// (K=16's refine went 89 s -> 874 s on allocation churn alone, docs/CHORD_GNSS_MS_SPLIT_
    /// SEARCH.md section 8) and fixed there; the ordinary refine still had it. Hoist the buffer
    /// out of the loop and the allocator is touched once per thread per pass instead.
    void hoprate_stream_into(const HopRateFilter& f, int p, long long window_start_sample,
                             double code_phase_chips, double doppler_hz, int n_hops,
                             const std::function<float(long long)>& nav_bit, int nh_phase,
                             std::vector<std::vector<std::complex<float>>>& out) const;

    /// Global channel indices whose passband covers the carrier at @c doppler_hz.
    std::vector<int> covering_bins(double doppler_hz, double doppler_margin_hz) const;

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

    /// Absolute code advance, in COMBINED chips mod @c eff_code_length, from sample 0 to the
    /// per-hop reference of the window starting at @c window_start_sample, at @c doppler_hz.
    /// This is the term every replica generator here adds internally (C(n) = CM*arg + n*cps).
    double window_advance_chips(long long window_start_sample, double doppler_hz) const;

    /// PHASE <-> ARGUMENT, the conversion every cross-stage code phase must go through.
    ///
    /// Every generator in this file (and the GPU kernel that mirrors them) defines the code
    /// phase at ABSOLUTE sample n as C(n) = comb_mult*code_phase_chips + n*cps(doppler). The
    /// `code_phase_chips` ARGUMENT is therefore not a physical phase: it is a phase
    /// back-referenced to sample 0 along a Doppler-scaled code rate, and at CHORD's uptime
    /// (n ~ 1.9e15 samples, 6.8 days) that back-reference has a lever of
    ///
    ///     d(argument)/d(doppler) = -n * chip_rate/(sample_rate * carrier) ~ 5095 chips per Hz
    ///
    /// measured 2026-07-31 by injection (509.50 chips at 0.1 Hz, predicted 509.49). So an
    /// argument is only meaningful ALONGSIDE the exact Doppler it was derived at: hand one to a
    /// consumer that uses a Doppler 0.002 Hz different and the despread lands 10 chips away,
    /// outside the DLL's capture, with nothing in the numbers to say so. That is why the search
    /// -> broker -> tracker seed never locked.
    ///
    /// The rule these two functions exist to enforce: NEVER transport or extrapolate an
    /// argument. Convert it to a physical phase at its own epoch with its OWN Doppler
    /// (@c phase_from_arg -- the lever cancels exactly), do all propagation in the phase
    /// domain, and convert back with the Doppler actually being passed to the generator
    /// (@c arg_from_phase). A Doppler error then only accrues over the propagation interval
    /// (seconds), not over the uptime (days).
    ///
    /// @c phase is in combined chips mod @c eff_code_length; @c arg is in the generator's own
    /// units mod @c code_length (identical unless comb_mult > 1).
    double phase_from_arg(double arg, long long window_start_sample, double doppler_hz) const;
    double arg_from_phase(double phase, long long window_start_sample, double doppler_hz) const;

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
    /// Truncate the per-hop chip gather to at most this many chips (0 = the filter's true
    /// span). Synthesis is LINEAR in the gather depth and is 89% of the tracker's GPU kernel,
    /// so this is the biggest single lever on GPU load -- and it is NOT free: the PFB span is
    /// the channel response, so a shorter gather is a DIFFERENT replica. Set it here rather
    /// than on the GPU so the CPU despread, the search refine and the e2e harness all see the
    /// same replica and the accuracy cost is measurable at the real geometry.
    void set_max_chips(int n) { _max_chips = n; }
    int max_chips() const { return _max_chips; }

    long eff_code_length() const { return _eff_code_length; }
    int comb_mult() const { return _comb_mult; }
    const std::vector<int8_t>& full_code(int p) const { return _full_code[(size_t)p]; }

private:
    int _max_chips = 0; ///< 0 = the filter's true span (see set_max_chips)
    SignalDescriptor _sig;
    double _sample_rate;
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
