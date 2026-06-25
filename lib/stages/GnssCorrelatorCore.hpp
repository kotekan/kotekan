/**
 * @file
 * @brief Single-thread GNSS correlation core shared by the search and track stages.
 */

#ifndef GNSS_CORRELATOR_CORE_HPP
#define GNSS_CORRELATOR_CORE_HPP

#include "gnssSignal.hpp" // for SignalDescriptor

#include <complex>
#include <fftw3.h>
#include <vector>

namespace gnss {

/**
 * @class CorrelatorCore
 * @brief FFTW plans, per-PRN code FFTs, and the parallel-code-phase correlation
 *        primitives for ONE periodic-code signal -- the DSP shared by
 *        @ref GnssAcquire (search) and @ref GnssTracker (track).
 *
 * The math is identical to GpsReplicaCorrelator's: analytic-convert a real
 * block, carrier-wipe at a trial Doppler, then matched-filter by FFT
 * (@c IFFT(FFT(data) * conj(FFT(code)))). The split here is so the data FFT can
 * be computed once per Doppler bin and reused across many PRNs (the search's
 * batch hoist), while a single PRN over a few bins (tracking) uses the same
 * calls.
 *
 * @warning Not thread-safe. FFTW plan execution writes the plan's bound scratch
 * arrays, so each owning thread needs its own CorrelatorCore. Periodic codes
 * only (GPS L1 C/A, L2C CM, L5 I/Q); the time-assisted long pilot (L2C CL)
 * remains in GpsReplicaCorrelator.
 */
class CorrelatorCore {
public:
    /// @param sig periodic-code signal descriptor (throws if time_assisted)
    /// @param sample_rate real sample rate Fs out of the front end, Hz
    /// @param f_offset carrier offset in the digitized stream, Hz
    /// @param prns PRNs to build replicas for; correlate_into indexes into this
    CorrelatorCore(const SignalDescriptor& sig, double sample_rate, double f_offset,
                   const std::vector<int>& prns);
    ~CorrelatorCore();

    CorrelatorCore(const CorrelatorCore&) = delete;
    CorrelatorCore& operator=(const CorrelatorCore&) = delete;

    int Ns() const { return _Ns; }
    int n_prn() const { return (int)_prns.size(); }
    const SignalDescriptor& sig() const { return _sig; }

    /// Analytic-convert a real block (length Ns) to complex baseband (internal z).
    void to_analytic(const float* block);
    /// Carrier-wipe the analytic block at Doppler @p fd and forward-FFT it,
    /// caching the spectrum. Call once per Doppler bin; correlate_into() then
    /// reuses it for any PRN at that bin.
    void prepare_doppler(double fd);
    /// Correlate the prepared (Doppler-wiped) block against PRN index @p p:
    /// ADD |corr|^2 into @p mag2[Ns] (caller zeroes once per integration); if
    /// @p cpx is non-null, store this block's complex correlation there (for
    /// phase tracking). Must follow a prepare_doppler() for the desired bin.
    void correlate_into(int p, float* mag2, std::complex<float>* cpx);

private:
    SignalDescriptor _sig;
    double _sample_rate;
    double _f_offset;
    std::vector<int> _prns;
    int _Ns;

    // FFTW work buffers (length _Ns).
    fftwf_complex* z_time;  ///< real block as complex (analytic input)
    fftwf_complex* a_freq;  ///< analytic-conversion spectrum
    fftwf_complex* z;       ///< analytic (complex baseband) block
    fftwf_complex* d;       ///< carrier-wiped data
    fftwf_complex* Dsave;   ///< FFT(d), kept intact so it can be reused per PRN
    fftwf_complex* D;       ///< Dsave * conj(FFT(code)), the IFFT input
    fftwf_complex* corr;    ///< correlation output (time domain)

    fftwf_plan p_fwd_analytic; ///< z_time -> a_freq
    fftwf_plan p_inv_analytic; ///< a_freq -> z
    fftwf_plan p_fwd_data;     ///< d -> Dsave
    fftwf_plan p_inv_corr;     ///< D -> corr

    /// conj(FFT(code_replica)) per PRN, contiguous [_prns.size()][_Ns].
    std::vector<std::complex<float>> code_fft_conj;
};

} // namespace gnss

#endif // GNSS_CORRELATOR_CORE_HPP
