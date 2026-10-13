/**
 * @file
 * @brief Channelized GNSS replicas: a PRN's replica analysed through the F-engine's PFB.
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
 * @brief Generates a GNSS replica and analyses it through the same r2c PFB the F-engine
 *        uses, returning per-channel hop streams.
 *
 * The replica is a real passband @c code*cos(carrier) pushed through the @ref fftwEngine PFB
 * (real-input r2c, @c 2N real samples per hop, @c spectrum_length positive-frequency channels
 * in natural order). It is anchored to the absolute sample index, so its inter-channel phases
 * match the channelized data's exactly. Owns its FFTW plan and buffers and the expanded
 * spreading codes; not thread-safe (one bank per worker thread).
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

    /// Replica for PRN index @c p over @c n_hops hops from @c window_start_sample at the given
    /// code phase and Doppler; returns [spectrum_length][n_hops].
    /// @c nh_phase aligns the secondary (Neuman-Hofman) overlay: the +-1 overlay chip for
    /// absolute primary-period index @c k is @c secondary[(k + nh_phase) mod len]. A negative
    /// value (the default) leaves the overlay off, giving the raw despread in which
    /// @ref overlay_wipe searches for the alignment. No effect for signals without a secondary
    /// code (L1 C/A, L2C).
    std::vector<std::vector<std::complex<float>>> channels(int p, long long window_start_sample,
                                                           double code_phase_chips,
                                                           double doppler_hz, int n_hops,
                                                           int nh_phase = -1);

    /// Hop-rate channelized replica for the channels in @c want: equal to @ref channels() to
    /// machine precision, but built per chip rather than per sample. The code is constant over
    /// a chip, so the polyphase filter collapses to a per-chip sum: a chip contributes the
    /// cumulative-filter difference @c Phi[k_hi+1]-Phi[k_lo] over its integer tap range, which
    /// is exact because chip boundaries fall between samples. The carrier is a per-hop phasor
    /// plus the channel-carrier offset built into the filter; both images are kept. Each hop
    /// costs O(n_chips) multiply-adds; the O(num_taps*fft_len) filter build amortizes over a
    /// stream.
    ///
    /// @c nav_bit (optional) gives the +-1 data bit for an absolute chip index. Data-bit edges
    /// fall on code-period boundaries, so applying it per chip gives the exact data-wiped
    /// replica; a per-hop multiply would smear the ~num_taps hops that straddle an edge.
    /// Returns [want.size()][n_hops].
    std::vector<std::vector<std::complex<float>>>
    channels_hoprate(int p, long long window_start_sample, double code_phase_chips,
                     double doppler_hz, int n_hops, const std::vector<int>& want,
                     const std::function<float(long long)>& nav_bit = {}, int nh_phase = -1) const;

    /// The slowly varying half of the hop-rate generator: the cumulative channel filters (both
    /// carrier images) for @c chans at @c doppler_hz. Doppler enters only through the carrier
    /// offset, so a filter stays usable for tens of Hz of drift. Built by @ref hoprate_filter,
    /// cached by @ref HopRateReplicaStream, consumed by @ref hoprate_stream.
    struct HopRateFilter {
        double doppler_hz = 0.0;
        int prn_index = -1; ///< PRN this filter's carrier was built for (FDMA); -1 = band-wide
        int n_chips = 0;
        /// First chip of the gather window; chips below it contribute zero. Non-zero only with
        /// centered truncation (@ref set_chips_centered). The prototype is a windowed sinc with
        /// its main lobe mid-span, so a one-sided cap [0, max_chips) cuts through the peak: at
        /// CHORD's 210-chip span a cap of 105 keeps 50.6% of the filter energy, while the
        /// centered window [d_first, n_chips) of the same length keeps 99.6%.
        int d_first = 0;
        std::vector<int> chans;
        std::vector<std::vector<std::complex<double>>> PhiA, PhiB; ///< [channel][Lf+1]
        /// The carrier this filter was built at, radians/sample: the PRN's carrier offset plus
        /// the Doppler.
        double wc_built = 0.0;
    };
    /// @c prn_index selects that PRN's carrier offset (FDMA); -1 = the band offset alone. Under
    /// FDMA a filter is valid only for the PRN it was built for; @ref hoprate_stream asserts it.
    HopRateFilter hoprate_filter(const std::vector<int>& want, double doppler_hz,
                                 int prn_index = -1) const;

    /// @ref hoprate_filter's Phi tables written into caller storage as float (re, im) pairs,
    /// laid out [channel][Lf+1] like the GPU tables. Each entry is the double-precision prefix
    /// sum rounded once, so no complex<double> tables are allocated. @c phiA and @c phiB each
    /// hold want.size() * (Lf+1) pairs. Returns the filter's chip span and carrier, with
    /// @c chans, @c PhiA and @c PhiB left empty.
    HopRateFilter hoprate_filter_f32(const std::vector<int>& want, double doppler_hz, int prn_index,
                                     float* phiA, float* phiB) const;

    /// Stream @c n_hops from a prebuilt @ref HopRateFilter. The per-hop carrier phasor and code
    /// phase use the given @c code_phase_chips and @c doppler_hz exactly; the filter shape comes
    /// from @c f, built for a nearby Doppler. @c nav_bit is applied per chip, as in
    /// @ref channels_hoprate.
    std::vector<std::vector<std::complex<float>>>
    hoprate_stream(const HopRateFilter& f, int p, long long window_start_sample,
                   double code_phase_chips, double doppler_hz, int n_hops,
                   const std::function<float(long long)>& nav_bit = {}, int nh_phase = -1) const;

    /// @ref hoprate_stream into a caller-owned buffer, resized only when it does not fit, so a
    /// caller evaluating many trial phases in a parallel loop allocates once per thread rather
    /// than once per evaluation.
    void hoprate_stream_into(const HopRateFilter& f, int p, long long window_start_sample,
                             double code_phase_chips, double doppler_hz, int n_hops,
                             const std::function<float(long long)>& nav_bit, int nh_phase,
                             std::vector<std::vector<std::complex<float>>>& out) const;

    /// Global channel indices whose passband covers the carrier at @c doppler_hz.
    /// @c prn_index selects that PRN's carrier offset (FDMA); -1 = the band offset alone.
    std::vector<int> covering_bins(double doppler_hz, double doppler_margin_hz,
                                   int prn_index = -1) const;

    // ---- FDMA (GLONASS L1OF/L2OF) ----
    // CDMA signals share one carrier and identify the satellite by its code. GLONASS FDMA
    // satellites share one 511-chip code and are separated by carrier: satellite k sits at
    // 1246.0 + k*0.4375 MHz on L2, and the whole comb (k = -7..+6, 5.69 MHz) fits in one tuning.
    // These set a per-PRN carrier offset on top of @c f_offset; an unset PRN gets 0 Hz (CDMA).

    /// Extra carrier offset for PRN index @c p, Hz, on top of @c f_offset (0 = CDMA default).
    void set_prn_freq_offset(int prn_index, double df_hz);

    /// Whole table at once, indexed by PRN index (construction order); an empty vector clears
    /// it (CDMA). Stages pass the config's `prn_freq_offset_hz` here.
    void set_prn_freq_offsets(const std::vector<double>& df_hz);

    /// Number of PRNs with a non-zero offset, for the stage's startup log: an all-zero table on
    /// an FDMA signal puts every satellite at band centre, which looks like a broken code.
    int prn_freq_offsets_set() const;

    /// Total carrier offset from band centre for PRN index @c p: @c f_offset + its FDMA
    /// offset. @c p < 0 (or unset) gives the band offset alone.
    double carrier_offset(int prn_index) const;

    /// Covering channels for the union of every PRN's carrier, for a caller that extracts one
    /// channel set for all PRNs (the search does). Equal to @ref covering_bins for CDMA.
    /// Over-covering is safe: each PRN's replica is ~0 outside its own lobe, so the matched
    /// filter, which normalises by |replica|, gives those channels no weight. The cost is
    /// compute, not SNR.
    std::vector<int> covering_bins_union(double doppler_hz, double doppler_margin_hz) const;

    /// Bipolar code chip for PRN index @c p at fractional chip phase (wraps period).
    int8_t code_chip(int p, double chip_phase) const;

    // ---- Slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md) ----
    // The bank is the record of which PRN each slot holds; consumers ask it rather than keep
    // their own copy.

    /// The PRN in each slot, in construction order (@c code_chip / @c channels index into it).
    const std::vector<int>& prns() const {
        return _prns;
    }
    /// PRN in slot @c p; -1 if @c p is out of range.
    int prn_at(int p) const {
        return (p >= 0 && p < (int)_prns.size()) ? _prns[(size_t)p] : -1;
    }

    /// Repoint slot @c p at a different satellite in place, rebuilding only that slot's
    /// expanded code table. It has to be in place: @ref GnssCudaDespread holds the bank by
    /// reference, so the bank cannot be replaced under a live GPU command. Nothing else in the
    /// bank (signal, sample rate, PFB, comb_mult, code length) depends on which PRN a slot holds.
    ///
    /// This is not the whole swap. Everything keyed to slot @c p elsewhere (the GPU code table,
    /// Phi caches, seeds, trims, power averages, element cal, detections) still describes the
    /// old satellite, and the caller must reset it in the same step; see
    /// cudaGnssChordTrackState::apply_prn_swaps.
    ///
    /// Returns false, changing nothing, if @c p is out of range, @c prn has no code for this
    /// signal, or the bank has FDMA offsets set. Costs one code expansion.
    bool set_prn(int p, int prn);

    /// Secondary (Neuman-Hofman) overlay length in primary periods (NH10=10 on L5 I5,
    /// NH20=20 on L5 Q5); 0 if the signal has no overlay. Also the number of @c nh_phase
    /// alignments to search.
    int secondary_length() const {
        return _secondary_length;
    }

    /// Bipolar (+-1) secondary-overlay chip for absolute primary-period index @c period at
    /// alignment @c nh_phase; +1 if the signal has no secondary code (so it composes as a no-op).
    int overlay_sign(long long period, int nh_phase) const;

    /// Code-Doppler feed-forward sign: +1 means an approaching satellite's code runs fast, like
    /// its carrier; 0 disables the feed-forward. Set from config `code_doppler_sign`.
    double code_doppler_sign = 1.0;

    /// Absolute code advance, in COMBINED chips mod @c eff_code_length, from sample 0 to the
    /// per-hop reference of the window starting at @c window_start_sample, at @c doppler_hz.
    /// This is the term every replica generator here adds internally (C(n) = CM*arg + n*cps).
    double window_advance_chips(long long window_start_sample, double doppler_hz) const;

    /// Convert between a physical code phase and a generator argument. Every code phase that
    /// crosses a stage boundary goes through these.
    ///
    /// Every generator here, and the GPU kernel that mirrors them, defines the code phase at
    /// absolute sample n as C(n) = comb_mult*code_phase_chips + n*cps(doppler). The
    /// @c code_phase_chips argument is therefore not a physical phase: it is referred back to
    /// sample 0 along a Doppler-scaled code rate, with a lever of
    ///
    ///     d(argument)/d(doppler) = -n * chip_rate/(sample_rate * carrier)
    ///
    /// which is ~5095 chips per Hz at n ~ 1.9e15 samples (6.8 days of F-engine uptime). An
    /// argument is meaningful only with the exact Doppler it was derived at: with a Doppler
    /// 0.002 Hz different, the despread lands 10 chips away, outside the DLL's capture.
    ///
    /// So never transport or extrapolate an argument. Convert it to a phase at its own epoch
    /// with its own Doppler (@c phase_from_arg; the lever cancels exactly), propagate in the
    /// phase domain, and convert back with the Doppler actually passed to the generator
    /// (@c arg_from_phase). A Doppler error then accrues only over the propagation interval.
    ///
    /// @c phase is in combined chips mod @c eff_code_length; @c arg is in the generator's own
    /// units mod @c code_length (the same unless comb_mult > 1).
    double phase_from_arg(double arg, long long window_start_sample, double doppler_hz) const;
    double arg_from_phase(double phase, long long window_start_sample, double doppler_hz) const;

    int spectrum_length() const {
        return _N;
    }
    int fft_len() const {
        return _fft_len;
    }
    double f_offset() const {
        return _f_offset;
    }
    int repl_period_hops() const {
        return _repl_period_hops;
    }
    double chip_rate_hz() const {
        return _sig.chip_rate_hz;
    }
    double carrier_hz() const {
        return _sig.carrier_hz;
    } ///< sky carrier (for code-Doppler)
    long code_length() const {
        return _sig.code_length;
    }
    // Combined-stream (TDM/BOC expanded) quantities and the code table, which an external (GPU)
    // despread needs to reproduce hoprate_stream exactly (see GnssCudaDespread).
    double eff_chip_rate() const {
        return _eff_chip_rate;
    }
    /// Truncate the per-hop chip gather to at most this many chips (0 = the filter's full
    /// span). Synthesis cost is linear in the gather depth, but the PFB span is the channel
    /// response, so a shorter gather is a different replica. Set on the bank so the CPU
    /// despread and the search refine build the same replica as the GPU path.
    void set_max_chips(int n) {
        _max_chips = n;
    }
    int max_chips() const {
        return _max_chips;
    }
    /// Centre the max_chips window on the prototype peak instead of starting it at chip 0
    /// (see HopRateFilter::d_first).
    void set_chips_centered(bool c) {
        _chips_centered = c;
    }

    long eff_code_length() const {
        return _eff_code_length;
    }
    int comb_mult() const {
        return _comb_mult;
    }
    const std::vector<int8_t>& full_code(int p) const {
        return _full_code[(size_t)p];
    }

private:
    /// The chip span, truncation and carrier shared by @ref hoprate_filter and
    /// @ref hoprate_filter_f32.
    HopRateFilter hoprate_header(double doppler_hz, int prn_index) const;
    /// Expand one PRN's component code into the combined stream this bank works in (TDM
    /// zero-stuffing, BOC half-cycles, or plain). Shared by the constructor and @ref set_prn so
    /// a slot swapped at runtime gets exactly the table it would have had at startup.
    std::vector<int8_t> build_full_code(int prn) const;

    int _max_chips = 0;           ///< 0 = the filter's true span (see set_max_chips)
    bool _chips_centered = false; ///< see set_chips_centered
    SignalDescriptor _sig;
    double _sample_rate;
    std::vector<double> _prn_df; ///< per-PRN FDMA carrier offset, Hz (empty/0 = CDMA)
    double _f_offset;
    int _N;
    int _fft_len; ///< 2*N real samples per hop (r2c)
    int _num_taps;
    int _repl_period_hops; ///< code_samples / gcd(fft_len, code_samples)
    // The bank works on the combined stream: a time-multiplexed component (L2C CM/CL) at its
    // tdm_phase parity with zeros at the sibling's, or a BOC(m,m) code expanded into 2m
    // half-cycles per chip. comb_mult = 1 (eff_* == _sig.*) for ordinary signals.
    int _comb_mult;        ///< 2 if time-multiplexed, 2m for BOC(m,m), else 1
    double _eff_chip_rate; ///< _sig.chip_rate_hz * _comb_mult (combined chipping rate)
    long _eff_code_length; ///< _sig.code_length * _comb_mult (combined-stream length)
    std::vector<float> _proto;
    std::vector<int> _prns; ///< PRN per slot; mutable via @ref set_prn
    std::vector<std::vector<int8_t>> _full_code;
    std::vector<int8_t> _secondary; ///< +-1 Neuman-Hofman overlay (NH10/NH20); empty if none
    int _secondary_length = 0;      ///< overlay period in primary periods; 0 = no overlay

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
    /// @param refresh_hz  rebuild when |doppler - built| exceeds this. It bounds the filter's
    ///                    staleness error (~-54 dB at 20 Hz, roughly linear). Rebuilds are
    ///                    cheap against Doppler drift of ~1 Hz/s, so a few Hz is a good value.
    HopRateReplicaStream(const ChannelizedReplicaBank& bank, int prn_index,
                         std::vector<int> channels, double refresh_hz = 2.0);

    /// Next @c n_hops from @c window_start at code phase @c cp and carrier @c doppler_hz.
    /// Rebuilds the cached filter on a large Doppler step; otherwise reuses it. @c nav_bit
    /// applies the +-1 data per chip (exact wipe). Returns [channels][n_hops] (owned here).
    const std::vector<std::vector<std::complex<float>>>&
    generate(long long window_start, double cp, double doppler_hz, int n_hops,
             const std::function<float(long long)>& nav_bit = {});

    /// Filter rebuilds so far (diagnostics; ~0 in steady lock).
    long rebuilds() const {
        return _rebuilds;
    }

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
