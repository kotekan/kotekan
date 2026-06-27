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

#include <complex> // for complex
#include <cstdint> // for int8_t
#include <fftw3.h> // for fftwf_plan, fftwf_complex
#include <vector>  // for vector

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
    std::vector<std::vector<std::complex<float>>>
    channels(int p, long long window_start_sample, double code_phase_chips, double doppler_hz,
             int n_hops);

    /// Hop-rate channelized replica for the listed channels -- numerically equal to
    /// @ref channels() but built per chip, not per sample. The spreading code is constant
    /// over a chip (~Fs/chip_rate samples), so the polyphase filter collapses to a per-chip
    /// sum: precompute the filter integrated over each chip vs sub-chip phase (the phi-bank,
    /// built once for this Doppler), then each hop is O(n_chips) MACs instead of
    /// O(num_taps*fft_len). Carrier = a per-hop output phasor + the channel-carrier offset
    /// baked into the (slowly-rebuilt) filter; both carrier images are kept. The phi-bank
    /// cost amortizes over a long stream, so this wins for streaming generation, not short
    /// bursts. Returns [want.size()][n_hops]. @c n_phi sub-chip phase bins (accuracy knob).
    std::vector<std::vector<std::complex<float>>>
    channels_hoprate(int p, long long window_start_sample, double code_phase_chips,
                     double doppler_hz, int n_hops, const std::vector<int>& want, int n_phi = 4096);

    /// Global channel indices whose passband covers the carrier at @c doppler_hz.
    std::vector<int> covering_bins(double doppler_hz, double doppler_margin_hz) const;

    /// Bipolar code chip for PRN index @c p at fractional chip phase (wraps period).
    int8_t code_chip(int p, double chip_phase) const;

    /// Code-Doppler feed-forward sign (+1 nominal: approaching -> faster code, as the
    /// carrier). Flip to -1 if the r2c fold inverts it vs the seed convention; set by
    /// the stage from config `code_doppler_sign` so it can be tuned without a rebuild.
    double code_doppler_sign = 1.0;

    int spectrum_length() const { return _N; }
    int fft_len() const { return _fft_len; }
    int repl_period_hops() const { return _repl_period_hops; }
    double chip_rate_hz() const { return _sig.chip_rate_hz; }
    double carrier_hz() const { return _sig.carrier_hz; } ///< sky carrier (for code-Doppler)
    long code_length() const { return _sig.code_length; }

private:
    SignalDescriptor _sig;
    double _sample_rate;
    double _f_offset;
    int _N;
    int _fft_len;   ///< 2*N real samples per hop (r2c)
    int _num_taps;
    int _repl_period_hops; ///< code_samples / gcd(fft_len, code_samples)
    std::vector<float> _proto;
    std::vector<std::vector<int8_t>> _full_code;

    float* _fold;
    fftwf_complex* _spec;
    fftwf_plan _p_fwd;
    std::vector<float> _replica_hist;
};

} // namespace gnss

#endif // GNSS_CHANNELIZED_REPLICA_HPP
