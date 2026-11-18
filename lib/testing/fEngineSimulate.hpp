#ifndef FENGINE_SIMULATE_HPP
#define FENGINE_SIMULATE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <fftw3.h>  // for fftwf_complex, fftwf_plan
#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class fEngineSimulate
 * @brief CPU reference F-engine: FFTs the windows crsWindowAlign assembles and quantizes the
 *        spectra into 4+4-bit CRS payload frames.
 *
 * The GPU implementation is verified against this stage. Per fft_window_job it acquires and
 * releases payload frames with the cursor protocol of fftWindowJob.hpp, gathers the window's
 * samples through the job's lookup table (which may point into the noise buffer), and emits one
 * output frame of the fft_length/2 quantized bins of the spectrum convention documented there:
 * x[n] = raw / 2048, X[k] = (2/fft_length) * DFT (DC kept, Nyquist dropped), bin k
 * stored as int4x2_swapped_withoffset of gain * X[k] (round half to even, clamp [-7, 7], real
 * in the high nibble). Output frames pair one-to-one and in order with crsHeaderGen's header
 * frames; ibvUdpSend consumes the pair by frame id.
 *
 * @par Buffers
 * @buffer job_buf         One fft_window_job + lookup table per frame, from crsWindowAlign
 *         @buffer_format fft_window_job_size(fft_length / samples_per_packet) bytes
 *         @buffer_metadata none
 * @buffer payload_buf     Payload slots from ibvUdpRecv; held and released per the job cursors
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 * @buffer noise_buf       crsWindowAlign's noise frame; taken once and held forever
 *         @buffer_format noise_slots slots of payload_slot bytes
 *         @buffer_metadata none
 * @buffer out_payload_buf One frame of fft_length/2 quantized bins per window, for ibvUdpSend
 *         @buffer_format fft_length/2 bytes
 *         @buffer_metadata none
 *
 * @conf fft_length         Int. Samples per window; a multiple of samples_per_packet.
 * @conf samples_per_packet Int, default 4096. 12-bit samples per input packet.
 * @conf gain               Float, default 1.0. Applied to every bin before quantization.
 * @conf bins_per_packet    Int, default 8192. Bins per output packet; must divide fft_length/2.
 *                          Used by crsHeaderGen, checked here for consistency.
 *
 * @author Andre Renard
 */
class fEngineSimulate : public kotekan::Stage {
public:
    fEngineSimulate(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~fEngineSimulate() override;
    void main_thread() override;

private:
    Buffer* job_buf;
    Buffer* payload_buf;
    Buffer* noise_buf;
    Buffer* out_buf;

    uint32_t fft_length;
    uint32_t samples_per_packet;
    uint32_t num_packets;
    uint32_t num_bins;
    float gain;

    float* samples = nullptr;
    fftwf_complex* spectrum = nullptr;
    fftwf_plan plan = nullptr;
};

#endif /* FENGINE_SIMULATE_HPP */
