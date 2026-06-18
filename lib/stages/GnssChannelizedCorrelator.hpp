/**
 * @file
 * @brief Measurement-tier channelized GNSS correlator.
 *  - GnssChannelizedCorrelator : public kotekan::Stage
 */

#ifndef GNSS_CHANNELIZED_CORRELATOR_HPP
#define GNSS_CHANNELIZED_CORRELATOR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "gnssSignal.hpp"      // for SignalDescriptor
#include "pfbPrototype.hpp"    // for Window
#include "restServer.hpp"      // for connectionInstance

#include <complex>  // for complex
#include <cstdint>  // for int8_t, uint8_t
#include <fftw3.h>  // for fftwf_complex, fftwf_plan
#include <mutex>    // for mutex
#include "json.hpp" // for json
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class GnssChannelizedCorrelator
 * @brief Despreads GNSS satellites directly in the PFB-channelized domain.
 *
 * The measurement tier of the distributed-band pipeline (see
 * @ref gnssChannelizedDespread.hpp). It consumes the channelized voltage stream
 * a PFB F-engine produces (@ref fftwEngine with num_taps>1: fftshifted
 * @c fftwf_complex spectra, @c spectrum_length channels per hop) and, for each
 * active PRN at its *known* code phase + Doppler, generates the matching
 * replica, PFB-analyzes it through the identical bank, gathers the channels
 * covering the carrier (@ref gnssBandPlan.hpp), and reports the complex despread
 * amplitude. Acquisition (finding code phase/Doppler) is assumed already done
 * and seeded here -- this stage only measures.
 *
 * The replica analysis must match the upstream F-engine exactly, so
 * @c spectrum_length, @c num_taps and @c pfb_window must equal the producer's.
 *
 * @par Buffers
 * @buffer in_buf Channelized voltages from the PFB F-engine.
 *     @buffer_format Array of @c fftwf_complex, [hops][spectrum_length], fftshifted
 *     @buffer_metadata none
 * @buffer out_buf One record per PRN per measurement window.
 *     @buffer_format float32[9] + float64 UTC per PRN (matches GpsReplicaCorrelator)
 *     @buffer_metadata none
 *
 * @conf signal            String (default "GPS_L1CA"). SignalDescriptor name.
 * @conf sample_rate       Double. Full (pre-channelization) sample rate, Hz.
 * @conf spectrum_length   Int. Channels per hop; must match the F-engine.
 * @conf num_taps          Int. PFB taps per channel; must match the F-engine.
 * @conf pfb_window        String (default "hamming"). Must match the F-engine.
 * @conf f_offset          Double (default 0). Carrier offset from band centre, Hz.
 * @conf prns              Array of Int. PRNs to measure.
 * @conf doppler_hz        Array of Double. Seeded Doppler per PRN (scalar broadcasts).
 * @conf code_phase_chips  Array of Double. Seeded code phase per PRN (scalar broadcasts).
 * @conf doppler_margin_hz Double (default 5000). Extra band for covering-channel selection.
 * @conf hops_per_record   Int (default: one code period). Coherent window length, hops.
 * @conf active_prns       Array of Int (optional). Initial go/no-go mask (default all on).
 * @conf capture_utc0      Double (default 0). UTC of sample 0; 0 = wall-clock at emit.
 *
 * Depends on libfftw3.
 *
 * @author Keith Vanderlinde
 */
class GnssChannelizedCorrelator : public kotekan::Stage {
public:
    GnssChannelizedCorrelator(kotekan::Config& config, const std::string& unique_name,
                              kotekan::bufferContainer& buffer_container);
    ~GnssChannelizedCorrelator() override;

    void main_thread() override;

private:
    void get_mask_callback(kotekan::connectionInstance& conn);
    void set_mask_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// Bipolar code chip for PRN @c p at fractional chip phase (wraps the period).
    int8_t code_chip(int p, double chip_phase) const;
    /// Generate + PFB-analyze the replica for PRN @c p over a window starting at
    /// @c window_start_sample; returns [spectrum_length][hops] fftshifted channels.
    std::vector<std::vector<std::complex<float>>> replica_channels(int p,
                                                                   long long window_start_sample);
    /// Channel indices whose passband covers the carrier for PRN @c p.
    std::vector<int> covering_bins(int p) const;

    static constexpr int RECORD_FLOATS = 11;
    static constexpr int RECORD_UTC_SLOT = 9;

    Buffer* in_buf;
    Buffer* out_buf;
    int frame_in;
    int frame_out;

    gnss::SignalDescriptor _sig;
    double _sample_rate;
    double _f_offset;
    double _capture_utc0;
    double _doppler_margin_hz;

    int _N;        ///< channels per hop (spectrum_length)
    int _num_taps; ///< PFB taps per channel
    dsp::Window _window_type;
    std::vector<float> _proto; ///< prototype matching the F-engine
    int _hops_per_record;

    std::vector<int> _prns;
    std::vector<double> _doppler;     ///< seeded Doppler per PRN, Hz
    std::vector<double> _code_phase;  ///< seeded code phase per PRN, chips
    std::vector<std::vector<int8_t>> _full_code; ///< cached code period per PRN

    std::vector<uint8_t> _active; ///< go/no-go mask aligned with _prns
    std::mutex _mask_mutex;

    // FFTW for the replica analysis (size _N forward, matches the F-engine).
    fftwf_complex* _fold;  ///< pfb_fold output (pre-FFT)
    fftwf_complex* _spec;  ///< FFT output (pre-fftshift)
    fftwf_plan _p_fwd;
    std::vector<std::complex<float>> _replica_hist; ///< persistent fold delay line
};

#endif // GNSS_CHANNELIZED_CORRELATOR_HPP
