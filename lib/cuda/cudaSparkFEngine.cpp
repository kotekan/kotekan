#include "cudaSparkFEngine.hpp"

#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "cudaUtils.hpp"         // for CHECK_CUDA_ERROR
#include "fEngineKernels.hpp"    // for gpu_unpack12, gpu_untangle_quantize
#include "fftWindowJob.hpp"      // for fft_window_job, fft_window_lut
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics
#include "tensorFFT.hpp"         // for gpu_tensor_fft
#include "visUtil.hpp"           // for frameID, current_time

#include "fmt.hpp" // for format

#include <complex>     // for polar
#include <cuda_fp16.h> // for __half2
#include <cufft.h>     // for cufftExecC2C, cufftPlan1d
#include <cufftXt.h>   // for cufftXtMakePlanMany, cufftXtExec
#include <functional>  // for bind
#include <stdexcept>   // for invalid_argument
#include <vector>      // for vector

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
    gain_buf(config.exists(unique_name, "gain_buf") ? get_buffer("gain_buf") : nullptr),
    stats_buf(config.exists(unique_name, "stats_buf") ? get_buffer("stats_buf") : nullptr),
    windows_counter(
        Metrics::instance().add_counter("kotekan_cuda_spark_fengine_windows_total", unique_name)),
    gpu_seconds_gauge(
        Metrics::instance().add_gauge("kotekan_cuda_spark_fengine_gpu_seconds", unique_name)) {

    fft_length = config.get<uint64_t>(unique_name, "fft_length");
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 4096);
    gain = config.get_default<float>(unique_name, "gain", 1.0f);
    gpu_id = config.get_default<int32_t>(unique_name, "gpu_id", 0);
    const std::string precision =
        config.get_default<std::string>(unique_name, "fft_precision", "fp32");
    if (precision != "fp32" && precision != "fp16")
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_precision must be \"fp32\" or \"fp16\", not \"{:s}\""),
                        unique_name, precision));
    fp16 = precision == "fp16";
    const std::string impl = config.get_default<std::string>(unique_name, "fft_impl", "cufft");
    if (impl != "cufft" && impl != "tensor")
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_impl must be \"cufft\" or \"tensor\", not \"{:s}\""),
                        unique_name, impl));
    tensor = impl == "tensor";
    fourstep = fp16 || tensor;
    if (tensor && fft_length > (1ull << 27))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: fft_impl tensor supports fft_length up to 2^27 (sub-FFTs up to 8192)"),
            unique_name));

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

    if (fourstep) {
        const size_t m = fft_length / 2;
        if ((fft_length & (fft_length - 1)) != 0 || samples_per_packet % 64 != 0 || m < 1024)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: the four-step paths (fp16 or fft_impl tensor) need a "
                                "power-of-two fft_length of at least 2^11 and "
                                "samples_per_packet a multiple of 64"),
                            unique_name));
        int log2m = 0;
        while ((1ull << log2m) < m)
            log2m++;
        m1 = 1ull << ((log2m + 1) / 2);
        m2 = m / m1;
    }

    stats_windows = config.get_default<uint32_t>(unique_name, "stats_windows", 64);
    if (gain_buf && gain_buf->frame_size < fft_length / 2 * sizeof(float))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: gain_buf frames hold {:d} bytes but a table is {:d}"),
                        unique_name, gain_buf->frame_size, fft_length / 2 * sizeof(float)));
    if (stats_buf && stats_buf->frame_size < fft_length / 2 * sizeof(float))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: stats_buf frames hold {:d} bytes but a table is {:d}"),
                        unique_name, stats_buf->frame_size, fft_length / 2 * sizeof(float)));

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
    if (gain_buf)
        gain_buf->register_consumer(unique_name);
    if (stats_buf)
        stats_buf->register_producer(unique_name);
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
    __half2* d_a = nullptr;
    __half2* d_b = nullptr;
    float2* d_a32 = nullptr;
    float2* d_b32 = nullptr;
    // The four-step twiddle indexes W^(2 (b k1 mod m)), up to order fft_length.
    const size_t hi_entries = (fft_length + 4095) / 4096;
    if (fourstep && fp16) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, m * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(__half2)));
    } else if (fourstep) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_a32, m * sizeof(float2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b32, m * sizeof(float2)));
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&d_time, fft_length * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_spec, m * sizeof(float2)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_lut, num_packets * sizeof(uint64_t)));
    float* d_gains = nullptr;
    float* d_stats = nullptr;
    if (gain_buf)
        CHECK_CUDA_ERROR(cudaMalloc(&d_gains, m * sizeof(float)));
    if (stats_buf) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_stats, m * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_stats, 0, m * sizeof(float)));
    }
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

    // One cuFFT plan over m for the plain fp32 path; batched fp16 sub-FFT plans over the
    // m1 x m2 four-step split for fp16 cufft; no plans at all with fft_impl tensor.
    cufftHandle plan = 0, plan2 = 0;
    if (fourstep && fp16 && !tensor) {
        long long n1 = m1, n2 = m2;
        size_t ws;
        CHECK_CUFFT(cufftCreate(&plan));
        CHECK_CUFFT(cufftXtMakePlanMany(plan, 1, &n1, nullptr, 1, n1, CUDA_C_16F, nullptr, 1, n1,
                                        CUDA_C_16F, m2, &ws, CUDA_C_16F));
        CHECK_CUFFT(cufftCreate(&plan2));
        CHECK_CUFFT(cufftXtMakePlanMany(plan2, 1, &n2, nullptr, 1, n2, CUDA_C_16F, nullptr, 1, n2,
                                        CUDA_C_16F, m1, &ws, CUDA_C_16F));
        CHECK_CUFFT(cufftSetStream(plan2, stream));
    } else if (!fourstep) {
        CHECK_CUFFT(cufftPlan1d(&plan, (int)m, CUFFT_C2C, 1));
    }
    if (plan)
        CHECK_CUFFT(cufftSetStream(plan, stream));
    INFO("FFT of {:d} points ({:d} packets) on GPU {:d}, {:s}{:s}, scale {:g}", fft_length,
         num_packets, gpu_id,
         fourstep ? (fp16 ? (tensor ? "fp16 tensor four-step " : "fp16 four-step ")
                          : "tf32 tensor four-step ")
                  : "fp32",
         fourstep ? fmt::format(fmt("{:d}x{:d}"), m1, m2) : "", 2.0 / (double)fft_length * gain);

    // The twiddle pass divides four-step intermediates by m1 (fp16 needs the range; the
    // tf32 path keeps the same semantics); fold it back here.
    const float scale = (float)(2.0 / (double)fft_length) * (fourstep ? (float)m1 : 1.0f);

    // Run every kernel and plan once on zeroed scratch: the first executions JIT and load
    // lazily, and that stall must land before packets are flowing, not on window one.
    {
        uint8_t* scratch = fourstep ? (fp16 ? (uint8_t*)d_a : (uint8_t*)d_a32) : (uint8_t*)d_time;
        std::vector<uint64_t> warm_lut(num_packets, (uint64_t)(uintptr_t)scratch);
        CHECK_CUDA_ERROR(cudaMemcpy(d_lut, warm_lut.data(), num_packets * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice));
        if (fourstep && fp16) {
            gpu_unpack12_fp16_t(d_lut, samples_per_packet, m1, m2, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m1, m2, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan, d_a, d_b, CUFFT_FORWARD));
            gpu_twiddle_transpose(d_b, m1, m2, d_whi, d_wlo, 1.0f / m1, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m2, m1, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan2, d_a, d_b, CUFFT_FORWARD));
            gpu_quantize_fourstep(d_b, m1, m2, d_whi, d_wlo, scale, gain, nullptr, nullptr,
                                  (uint8_t*)d_a, stream);
        } else if (fourstep) {
            gpu_unpack12_f32_t(d_lut, samples_per_packet, m1, m2, d_a32, stream);
            gpu_tensor_fft_f32(d_a32, d_b32, m1, m2, stream);
            gpu_twiddle_transpose_f32(d_b32, m1, m2, d_whi, d_wlo, 1.0f / m1, d_a32, stream);
            gpu_tensor_fft_f32(d_a32, d_b32, m2, m1, stream);
            gpu_quantize_fourstep_f32(d_b32, m1, m2, d_whi, d_wlo, scale, gain, nullptr, nullptr,
                                      (uint8_t*)d_a32, stream);
        } else {
            gpu_unpack12(d_lut, num_packets, samples_per_packet, d_time, stream);
            CHECK_CUFFT(
                cufftExecC2C(plan, (cufftComplex*)d_time, (cufftComplex*)d_spec, CUFFT_FORWARD));
            gpu_untangle_quantize(d_spec, m, d_whi, d_wlo, scale, gain, nullptr, nullptr,
                                  (uint8_t*)d_time, stream);
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    frameID job_id(job_buf), out_id(out_buf), acq_id(payload_buf), rel_id(payload_buf);
    uint64_t acquired = 0, released = 0;
    frameID gain_id(gain_buf ? gain_buf : job_buf), stats_id(stats_buf ? stats_buf : job_buf);
    uint64_t window_count = 0;

    // Hold processing until the first gain table, so no window goes out with the wrong gain.
    if (gain_buf) {
        const uint8_t* table = gain_buf->wait_for_full_frame(unique_name, gain_id);
        if (table == nullptr)
            return;
        CHECK_CUDA_ERROR(cudaMemcpy(d_gains, table, m * sizeof(float), cudaMemcpyHostToDevice));
        gain_buf->mark_frame_empty(unique_name, gain_id);
        gain_id++;
    }

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

        // Pick up a new gain table when one is waiting.
        if (gain_buf) {
            const auto now = double_to_ts(current_time());
            if (gain_buf->wait_for_full_frame_timeout(unique_name, gain_id, now) == 0) {
                CHECK_CUDA_ERROR(cudaMemcpy(d_gains, gain_buf->frames[gain_id], m * sizeof(float),
                                            cudaMemcpyHostToDevice));
                gain_buf->mark_frame_empty(unique_name, gain_id);
                gain_id++;
            }
        }
        uint8_t* stats_frame = nullptr;
        const bool emit_stats = stats_buf && (window_count + 1) % stats_windows == 0;
        if (emit_stats) {
            stats_frame = stats_buf->wait_for_empty_frame(unique_name, stats_id);
            if (stats_frame == nullptr)
                break;
        }

        const double start = current_time();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_lut, fft_window_lut(job), num_packets * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice, stream));
        if (fourstep && fp16) {
            gpu_unpack12_fp16_t(d_lut, samples_per_packet, m1, m2, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m1, m2, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan, d_a, d_b, CUFFT_FORWARD));
            gpu_twiddle_transpose(d_b, m1, m2, d_whi, d_wlo, 1.0f / m1, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m2, m1, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan2, d_a, d_b, CUFFT_FORWARD));
            gpu_quantize_fourstep(d_b, m1, m2, d_whi, d_wlo, scale, gain, d_gains, d_stats,
                                  out_frame, stream);
        } else if (fourstep) {
            gpu_unpack12_f32_t(d_lut, samples_per_packet, m1, m2, d_a32, stream);
            gpu_tensor_fft_f32(d_a32, d_b32, m1, m2, stream);
            gpu_twiddle_transpose_f32(d_b32, m1, m2, d_whi, d_wlo, 1.0f / m1, d_a32, stream);
            gpu_tensor_fft_f32(d_a32, d_b32, m2, m1, stream);
            gpu_quantize_fourstep_f32(d_b32, m1, m2, d_whi, d_wlo, scale, gain, d_gains, d_stats,
                                      out_frame, stream);
        } else {
            gpu_unpack12(d_lut, num_packets, samples_per_packet, d_time, stream);
            CHECK_CUFFT(
                cufftExecC2C(plan, (cufftComplex*)d_time, (cufftComplex*)d_spec, CUFFT_FORWARD));
            gpu_untangle_quantize(d_spec, m, d_whi, d_wlo, scale, gain, d_gains, d_stats, out_frame,
                                  stream);
        }
        if (emit_stats) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(stats_frame, d_stats, m * sizeof(float),
                                             cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_stats, 0, m * sizeof(float), stream));
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        gpu_seconds_gauge.set(current_time() - start);
        window_count++;
        if (emit_stats) {
            stats_buf->mark_frame_full(unique_name, stats_id);
            stats_id++;
        }

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

    if (plan)
        CHECK_CUFFT(cufftDestroy(plan));
    if (plan2)
        CHECK_CUFFT(cufftDestroy(plan2));
    CHECK_CUDA_ERROR(cudaFree(fourstep ? (fp16 ? (void*)d_a : (void*)d_a32) : (void*)d_time));
    CHECK_CUDA_ERROR(cudaFree(fourstep ? (fp16 ? (void*)d_b : (void*)d_b32) : (void*)d_spec));
    CHECK_CUDA_ERROR(cudaFree(d_lut));
    if (d_gains)
        CHECK_CUDA_ERROR(cudaFree(d_gains));
    if (d_stats)
        CHECK_CUDA_ERROR(cudaFree(d_stats));
    CHECK_CUDA_ERROR(cudaFree(d_whi));
    CHECK_CUDA_ERROR(cudaFree(d_wlo));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
