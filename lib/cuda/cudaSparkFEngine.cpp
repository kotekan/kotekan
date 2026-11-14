#include "cudaSparkFEngine.hpp"

#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "cudaUtils.hpp"         // for CHECK_CUDA_ERROR
#include "fEngineKernels.hpp"    // for gpu_unpack12, gpu_untangle_quantize
#include "fftWindowJob.hpp"      // for fft_window_job, fft_window_lut
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "visUtil.hpp"           // for frameID, current_time

#include "fmt.hpp" // for format

#include <complex>    // for polar
#include <cufft.h>    // for cufftExecC2C, cufftPlan1d
#include <functional> // for bind
#include <stdexcept>  // for invalid_argument
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(cudaSparkFEngine);

#define CHECK_CUFFT(cmd)                                                                           \
    do {                                                                                           \
        const cufftResult result = (cmd);                                                          \
        if (result != CUFFT_SUCCESS)                                                               \
            FATAL_ERROR("cuFFT error at {:s}:{:d}: {:d}", __FILE__, __LINE__, (int)result);        \
    } while (0)

cudaSparkFEngine::cudaSparkFEngine(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&cudaSparkFEngine::main_thread, this)),
    job_buf(get_buffer("job_buf")), payload_buf(get_buffer("payload_buf")),
    noise_buf(get_buffer("noise_buf")), out_buf(get_buffer("out_payload_buf")),
    windows_counter(
        Metrics::instance().add_counter("kotekan_cuda_spark_fengine_windows_total", unique_name)),
    gpu_seconds_gauge(
        Metrics::instance().add_gauge("kotekan_cuda_spark_fengine_gpu_seconds", unique_name)) {

    fft_length = config.get<uint64_t>(unique_name, "fft_length");
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 4096);
    gain = config.get_default<float>(unique_name, "gain", 1.0f);
    gpu_id = config.get_default<int32_t>(unique_name, "gpu_id", 0);

    if (fft_length == 0 || fft_length % samples_per_packet != 0 || fft_length > (1ull << 28))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_length {:d} must be a multiple of samples_per_packet "
                            "{:d} and at most 2^28"),
                        unique_name, fft_length, samples_per_packet));
    num_packets = fft_length / samples_per_packet;
    if (out_buf->frame_size < fft_length / 2)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: out_payload_buf frames hold {:d} bytes but a spectrum is "
                            "{:d}"),
                        unique_name, out_buf->frame_size, fft_length / 2));
    if (job_buf->frame_size < fft_window_job_size(num_packets))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: job_buf frames hold {:d} bytes but a window of {:d} packets "
                "needs {:d}"),
            unique_name, job_buf->frame_size, num_packets, fft_window_job_size(num_packets)));

    device = cudaDeviceInterface::get(gpu_id, unique_name, config);
    int host_page_tables = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
        &host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, gpu_id));
    if (!host_page_tables)
        FATAL_ERROR("GPU {:d} cannot read pageable host memory through the host page tables, "
                    "which cudaSparkFEngine needs to read the receive buffers in place",
                    gpu_id);

    job_buf->register_consumer(unique_name);
    payload_buf->register_consumer(unique_name);
    noise_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

void cudaSparkFEngine::main_thread() {
    // The aligner fills this once at startup; the lookup tables point into it for lost packets.
    if (noise_buf->wait_for_full_frame(unique_name, 0) == nullptr)
        return;

    device->set_thread_device();

    const size_t m = fft_length / 2;
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float* d_time = nullptr;
    float2* d_spec = nullptr;
    uint64_t* d_lut = nullptr;
    float2* d_whi = nullptr;
    float2* d_wlo = nullptr;
    const size_t hi_entries = (m + 4095) / 4096;
    CHECK_CUDA_ERROR(cudaMalloc(&d_time, fft_length * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_spec, m * sizeof(float2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_lut, num_packets * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_whi, hi_entries * sizeof(float2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_wlo, 4096 * sizeof(float2)));

    // W = exp(-2 pi i / fft_length); W^k = w_hi[k >> 12] * w_lo[k & 4095], tables in double.
    {
        std::vector<float2> whi(hi_entries), wlo(4096);
        const double step = -2.0 * M_PI / (double)fft_length;
        for (size_t j = 0; j < hi_entries; j++) {
            const std::complex<double> w = std::polar(1.0, step * (double)(j * 4096));
            whi[j] = make_float2((float)w.real(), (float)w.imag());
        }
        for (size_t j = 0; j < 4096; j++) {
            const std::complex<double> w = std::polar(1.0, step * (double)j);
            wlo[j] = make_float2((float)w.real(), (float)w.imag());
        }
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_whi, whi.data(), hi_entries * sizeof(float2), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_wlo, wlo.data(), 4096 * sizeof(float2), cudaMemcpyHostToDevice));
    }

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, (int)m, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftSetStream(plan, stream));
    INFO("FFT of {:d} points ({:d} packets) on GPU {:d}, scale {:g}", fft_length, num_packets,
         gpu_id, 2.0 / (double)fft_length * gain);

    const float scale = (float)(2.0 / (double)fft_length) * gain;
    frameID job_id(job_buf), out_id(out_buf), acq_id(payload_buf), rel_id(payload_buf);
    uint64_t acquired = 0, released = 0;

    while (!stop_thread) {
        uint8_t* job_frame = job_buf->wait_for_full_frame(unique_name, job_id);
        if (job_frame == nullptr)
            break;
        const fft_window_job* job = reinterpret_cast<const fft_window_job*>(job_frame);
        if (job->num_packets != num_packets)
            FATAL_ERROR("Job carries {:d} packets per window, this stage expects {:d}",
                        job->num_packets, num_packets);

        bool shutdown = false;
        while (acquired < job->acquire_up_to) {
            if (payload_buf->wait_for_full_frame(unique_name, acq_id) == nullptr) {
                shutdown = true;
                break;
            }
            acq_id++;
            acquired++;
        }
        if (shutdown)
            break;
        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out_frame == nullptr)
            break;

        const double start = current_time();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lut, fft_window_lut(job), num_packets * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice, stream));
        gpu_unpack12(d_lut, num_packets, samples_per_packet, d_time, stream);
        CHECK_CUFFT(
            cufftExecC2C(plan, (cufftComplex*)d_time, (cufftComplex*)d_spec, CUFFT_FORWARD));
        gpu_untangle_quantize(d_spec, m, d_whi, d_wlo, scale, out_frame, stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        gpu_seconds_gauge.set(current_time() - start);

        while (released < job->release_up_to) {
            payload_buf->mark_frame_empty(unique_name, rel_id);
            rel_id++;
            released++;
        }
        job_buf->mark_frame_empty(unique_name, job_id);
        job_id++;
        out_buf->mark_frame_full(unique_name, out_id);
        out_id++;
        windows_counter.inc();
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_time));
    CHECK_CUDA_ERROR(cudaFree(d_spec));
    CHECK_CUDA_ERROR(cudaFree(d_lut));
    CHECK_CUDA_ERROR(cudaFree(d_whi));
    CHECK_CUDA_ERROR(cudaFree(d_wlo));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
