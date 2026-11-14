#ifndef CUDA_SPARK_FENGINE_HPP
#define CUDA_SPARK_FENGINE_HPP

#include "Config.hpp"              // for Config
#include "Stage.hpp"               // for Stage
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "prometheusMetrics.hpp"   // for Counter, Gauge

#include <memory>   // for shared_ptr
#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for string

/**
 * @class cudaSparkFEngine
 * @brief GPU F-engine: consumes fft_window_jobs, FFTs each window in place from the packet
 *        payloads, and quantizes the spectrum into the output payload buffer.
 *
 * A plain Stage calling CUDA directly rather than a cudaProcess: with the Spark's coherent
 * unified memory there are no staging copies to pipeline. The kernels read the packet
 * payloads and the noise slots through the job's lookup table straight out of the receive
 * buffers (host memory), and the quantize kernel writes straight into the output payload
 * frame. The stage owns a non-blocking stream of its own and device buffers for the
 * unpacked window and its spectrum; it synchronizes once per window.
 * The GPU must read pageable host memory through the host's page tables, as the Spark's
 * does; a discrete PCIe GPU cannot, and the stage refuses one at startup.
 *
 * Per window: unpack 12-bit samples to float32, one cuFFT C2C of fft_length/2 points over
 * the real-as-complex pairs, then the untangle + quantize kernel applies the spectrum
 * convention and gain of fftWindowJob.hpp and writes fft_length/2 offset-encoded 4+4-bit
 * bytes. Payload frames are held and released with the acquire/release cursors the job
 * carries (see fftWindowJob.hpp); one output frame is produced per job, in order, pairing
 * with crsHeaderGen's header frames.
 *
 * @par Buffers
 * @buffer job_buf          fft_window_jobs from crsWindowAlign
 *         @buffer_format fft_window_job + lookup table
 *         @buffer_metadata none
 * @buffer payload_buf      Receive payload slots; read in place through the lookup table
 *         @buffer_format packets_per_frame slots of payload_slot bytes
 *         @buffer_metadata none
 * @buffer noise_buf        Noise slots for lost packets; read through the lookup table
 *         @buffer_format noise_slots slots of payload_slot bytes
 *         @buffer_metadata none
 * @buffer out_payload_buf  One quantized spectrum per frame
 *         @buffer_format fft_length / 2 bytes, int4x2_swapped_withoffset
 *         @buffer_metadata none
 *
 * @conf fft_length         Int. Samples per window; a multiple of samples_per_packet, at
 *                          most 2^28.
 * @conf samples_per_packet Int, default 4096.
 * @conf gain               Float, default 1.0. Applied to every bin before quantization.
 * @conf gpu_id             Int, default 0.
 * @conf fft_precision      String, default "fp32". "fp16" runs the four-step path: the
 *                          fft_length/2-point transform becomes two batched fp16 cuFFTs
 *                          with a twiddle/transpose pass between (cuFFT applies no scaling
 *                          and overflows fp16 at these sizes in one piece; the twiddle pass
 *                          scales by 1/m1), halving the memory traffic. Needs fft_length a
 *                          power of two of at least 2^11 and samples_per_packet a multiple
 *                          of 64.
 *
 * @par Metrics
 * @metric kotekan_cuda_spark_fengine_windows_total  Windows processed.
 * @metric kotekan_cuda_spark_fengine_gpu_seconds    GPU time of the last window.
 *
 * @author Andre Renard
 */
class cudaSparkFEngine : public kotekan::Stage {
public:
    cudaSparkFEngine(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~cudaSparkFEngine() override = default;
    void main_thread() override;

private:
    Buffer* job_buf;
    Buffer* payload_buf;
    Buffer* noise_buf;
    Buffer* out_buf;

    uint64_t fft_length;
    uint32_t samples_per_packet;
    uint32_t num_packets;
    float gain;
    int32_t gpu_id;
    bool fp16;
    size_t m1 = 0, m2 = 0;

    std::shared_ptr<cudaDeviceInterface> device;

    kotekan::prometheus::Counter& windows_counter;
    kotekan::prometheus::Gauge& gpu_seconds_gauge;
};

#endif /* CUDA_SPARK_FENGINE_HPP */
