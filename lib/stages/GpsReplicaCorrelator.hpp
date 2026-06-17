/**
 * @file
 * @brief GPS L1 C/A replica correlator stage.
 *  - GpsReplicaCorrelator : public kotekan::Stage
 */

#ifndef GPS_REPLICA_CORRELATOR_HPP
#define GPS_REPLICA_CORRELATOR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <complex>  // for complex
#include <fftw3.h>  // for fftwf_complex, fftwf_plan
#include <stdint.h> // for int16_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class GpsReplicaCorrelator
 * @brief Correlate a single-antenna real-sampled Airspy stream against
 *        software-generated GPS L1 C/A replicas, one record per PRN.
 *
 * For each 1 ms code period the stage performs the standard parallel-code-phase
 * acquisition search of every configured PRN over a coarse Doppler grid, and
 * emits the complex correlation peak (amplitude, and phase when @c track_phase
 * is set). A constellation of GPS transits thus maps the antenna beam response
 * with the satellites as calibration point sources -- the replacement for
 * @c SimpleCrosscorr in the GPS use case, which cannot handle per-PRN Doppler.
 *
 * @par DSP outline (per 1 ms block, per PRN, per Doppler bin)
 *  -# Slice @c Ns = round(sample_rate/1000) real samples (assembled across
 *     input frames with a persistent carry buffer).
 *  -# Analytic-convert the real block to complex baseband (FFT -> zero the
 *     negative-frequency half, double the positive half -> IFFT). Done once
 *     per block; everything downstream is standard complex baseband.
 *  -# Wipe the carrier off the data: multiply by
 *     @c exp(-j*2*pi*(f_offset+fd)*t).
 *  -# Code correlation by FFT: @c IFFT(FFT(d) * conj(FFT(code_replica))).
 *  -# Record the peak magnitude, its lag (code phase) and the complex peak.
 *
 * The largest peak across the Doppler grid wins. When @c incoherent_ms > 1,
 * the per-(Doppler,lag) magnitude-squared surface is incoherently summed over
 * that many consecutive blocks before the peak is picked and a record emitted.
 *
 * @note Front-end frequency bookkeeping: @c airspyInput already subtracts the
 * ADC DC offset and multiplies by @c (-1)^n (an @c Fs/2 spectral shift folding
 * the band into the first Nyquist zone). @c f_offset must therefore be the
 * carrier location in that already-shifted real stream, not the raw tuner IF.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer.
 *     @buffer_format Array of @c int16_t real samples (Airspy, DC-corrected)
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer.
 *     @buffer_format float32 [num_prns, record_floats] per integration
 *     @buffer_metadata none
 *
 * @conf   sample_rate      Double (default 5e6). Real sample rate Fs, in Hz;
 *                          must equal the true rate out of @c airspyInput.
 * @conf   f_offset         Double (default 1e6). Known carrier offset of L1 in
 *                          the digitized stream, in Hz (see front-end note).
 * @conf   doppler_min      Double (default -6000). Doppler grid start, Hz.
 * @conf   doppler_max      Double (default  6000). Doppler grid end, Hz.
 * @conf   doppler_step     Double (default   500). Doppler grid step, Hz.
 * @conf   prns             Array of Int. Active PRN list (1..32). Required.
 * @conf   incoherent_ms    Int (default 1). 1 ms blocks summed (incoherently)
 *                          per emitted record.
 * @conf   track_phase      Bool (default false). Retain the complex peak phase
 *                          and run per-PRN phase tracking (Stage 2): nav-bit
 *                          stripping + continuous carrier phase. The amplitude
 *                          path is identical either way. Best used with
 *                          @c incoherent_ms == 1 (1 ms coherent blocks).
 * @conf   lock_snr         Double (default 15). SNR threshold (peak/mean of the
 *                          correlation surface) above which a PRN is considered
 *                          locked for phase tracking. Below it, nav_sign and
 *                          phase_cont are reported as 0/unlocked.
 *
 * Depends on libfftw3.
 *
 * @author Keith Vanderlinde
 */
class GpsReplicaCorrelator : public kotekan::Stage {
public:
    GpsReplicaCorrelator(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GpsReplicaCorrelator() override;

    void main_thread() override;

    /// Number of float32 *slots* per per-PRN output record. Field order:
    /// {prn, doppler_hz, code_phase_chips, peak_amp, peak_re, peak_im, snr,
    ///  nav_sign, phase_cont} as float32, then a trailing float64 UTC timestamp
    /// occupying the final two slots (slots 9-10; UNIX seconds don't fit a
    /// float32, so it is type-punned across two slots -- same single-array
    /// convention as the count word in make_power_corr_desc).
    /// nav_sign / phase_cont / peak_re / peak_im are populated only under
    /// @c track_phase. The UTC stamp is the wall-clock time at emit, shared by
    /// all PRNs in a record block.
    static constexpr int RECORD_FLOATS = 11;
    /// Slot index of the trailing float64 UTC timestamp.
    static constexpr int RECORD_UTC_SLOT = 9;

private:
    /// Assemble exactly @c _Ns real samples into @c block (handles input
    /// frames that don't align to the 1 ms code period). Returns false when
    /// the input stream has ended.
    bool fill_block();

    /// Analytic-convert the current real @c block into @c z (complex baseband).
    void to_analytic();

    /// Correlate the analytic block against PRN index @p p over Doppler bins
    /// [@p di_lo, @p di_hi] (inclusive), accumulating |corr|^2 into @c accum.
    /// A locked PRN narrows this window to a few bins around its tracked
    /// Doppler; an unlocked one sweeps the whole grid.
    void correlate_prn(int p, int di_lo, int di_hi);

    /// Stage 2 per-block phase tracking for PRN index @p p: gate on lock,
    /// strip nav-bit pi flips (de-rotated by the tracked carrier frequency),
    /// run a frequency-locked loop, and maintain the continuous carrier phase.
    /// @p doppler_bin is the grid bin used for wipe-off; @p doppler_seed is the
    /// parabola-refined acquisition Doppler used to seed a fresh lock. Writes
    /// @p nav_sign (cumulative +-1, 0 if unlocked), @p phase_cont (unwrapped
    /// nav-stripped phase, rad), and @p doppler_out (FLL-refined Doppler, Hz).
    void track_phase_update(int p, std::complex<float> raw, float snr, double doppler_bin,
                            double doppler_seed, double code_phase_chips, float& nav_sign,
                            float& phase_cont, float& doppler_out);

    /// Input real samples (int16) and output records (float32).
    Buffer* in_buf;
    Buffer* out_buf;

    int frame_in;
    int frame_out;

    // --- configuration ---
    double _sample_rate;
    double _f_offset;
    double _doppler_min, _doppler_max, _doppler_step;
    int _incoherent_ms;
    bool _track_phase;
    double _lock_snr;
    std::vector<int> _prns;

    // --- derived sizes ---
    int _Ns;                          ///< samples per 1 ms block = round(Fs/1000)
    std::vector<double> _doppler_grid; ///< trial Doppler frequencies, Hz

    // --- per-block assembly ---
    std::vector<float> block;  ///< current real block, length _Ns
    int block_fill;            ///< samples currently in @c block
    int16_t* in_local;         ///< current input frame
    int in_pos;                ///< read index within current input frame (samples)
    int in_samples_per_frame;

    // --- FFTW work buffers (length _Ns) ---
    fftwf_complex* z_time;     ///< real block loaded as complex (analytic input)
    fftwf_complex* a_freq;     ///< spectrum during analytic conversion
    fftwf_complex* z;          ///< analytic (complex baseband) block
    fftwf_complex* d;          ///< carrier-wiped data
    fftwf_complex* D;          ///< FFT(d), reused as corr-spectrum
    fftwf_complex* corr;       ///< correlation output (time domain)

    fftwf_plan p_fwd_analytic; ///< z_time -> a_freq (sign -1)
    fftwf_plan p_inv_analytic; ///< a_freq -> z      (sign +1)
    fftwf_plan p_fwd_data;     ///< d -> D           (sign -1)
    fftwf_plan p_inv_corr;     ///< D -> corr        (sign +1)

    /// conj(FFT(code_replica)) for each PRN, contiguous [_prns.size()][_Ns].
    std::vector<std::complex<float>> code_fft_conj;

    /// Incoherent accumulator: [_prns.size()][n_doppler][_Ns] of |corr|^2,
    /// summed over the integration window for peak finding.
    std::vector<float> accum;
    /// Most-recent block's complex correlation, same [prn][doppler][_Ns]
    /// layout. Read at the winning (Doppler, lag) bin to recover the complex
    /// peak phase (meaningful when incoherent_ms == 1; see header note).
    std::vector<std::complex<float>> last_corr;
    int blocks_in_accum; ///< blocks summed into @c accum so far

    // --- Stage 2 per-PRN phase-tracking state (one entry per PRN) ---
    std::vector<uint8_t> trk_locked;            ///< currently phase-locked?
    std::vector<float> trk_sign;                ///< cumulative nav-bit sign (+-1)
    std::vector<std::complex<float>> trk_prev;  ///< previous sign-corrected peak
    std::vector<double> trk_cont_phase;         ///< accumulated unwrapped phase, rad
    std::vector<double> trk_prev_wrapped;       ///< previous corrected wrapped phase, rad
    std::vector<double> trk_prev_doppler;       ///< previous block's Doppler grid bin, Hz
    std::vector<double> trk_prev_code_phase;    ///< previous block's code phase, chips
    std::vector<double> trk_doppler;            ///< FLL absolute Doppler estimate, Hz
};

#endif // GPS_REPLICA_CORRELATOR_HPP
