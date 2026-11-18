#include "cudaSparkFEngine.hpp"

#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "cudaUtils.hpp"         // for CHECK_CUDA_ERROR
#include "fEngineKernels.hpp"    // for gpu_unpack12, gpu_untangle_quantize
#include "fftWindowJob.hpp"      // for fft_window_job, fft_window_lut
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO
#include "pfbKernels.hpp"        // for pfb_taps, pfb_proto_shared
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
    const uint32_t taps = config.get_default<uint32_t>(unique_name, "pfb_taps", 1);
    if (taps != 1 && taps != 4)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: pfb_taps must be 1 or 4, not {:d}"), unique_name, taps));
    pfb = taps == 4;
    if (pfb && !fp16)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: pfb_taps 4 requires fft_precision fp16"), unique_name));
    tensor = impl == "tensor";
    fourstep = fp16 || tensor;
    fused = config.get_default<bool>(unique_name, "fused_quantize", false);
    if (fused && !(tensor && fp16))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: fused_quantize needs fft_impl tensor and fft_precision fp16"), unique_name));
    // The tensor kernels run sub-FFTs of up to 8192 points in one CTA. The fp16 FFT splits
    // longer ones over several CTAs (tensorFFT.hpp); the TF32 and fused ones do not.
    if (tensor && !fp16 && fft_length > (1ull << 27))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_impl tensor with fft_precision fp32 supports fft_length "
                            "up to 2^27 (sub-FFTs up to 8192)"),
                        unique_name));
    if (fused && fft_length > (1ull << 28))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fused_quantize supports fft_length up to 2^28 (second "
                            "sub-FFT up to 8192)"),
                        unique_name));

    if (fft_length == 0 || fft_length % samples_per_packet != 0 || fft_length > (1ull << 31))
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_length {:d} must be a multiple of samples_per_packet "
                            "{:d} and at most 2^31"),
                        unique_name, fft_length, samples_per_packet));
    num_packets = fft_length / samples_per_packet;
    // The PFB holds each window's payload frames three windows longer. Short of four
    // windows and two partial frames, the producer stalls on a frame the stage holds and
    // no further job arrives. Slots are at least the packed payload, so this undercounts.
    if (pfb) {
        const uint64_t window_bytes = (uint64_t)num_packets * (3 * samples_per_packet / 2);
        const uint64_t need =
            (4 * window_bytes + payload_buf->frame_size - 1) / payload_buf->frame_size + 2;
        if ((uint64_t)payload_buf->num_frames < need)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: pfb_taps 4 needs payload_buf to hold four windows and "
                                "two partial frames, at least {:d} frames, not {:d}"),
                            unique_name, need, payload_buf->num_frames));
    }
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
        if (tensor) {
            split1 = tensor_fft_split(m1);
            split2 = tensor_fft_split(m2);
        }
        if (fused && m2 < 256)
            throw std::invalid_argument(
                fmt::format(fmt("{:s}: fused_quantize needs the four-step split's m2 >= 256 "
                                "(fft_length >= 2^18)"),
                            unique_name));
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
    // The PFB keeps the last four windows' tables in rotating slots of d_lut.
    CHECK_CUDA_ERROR(cudaMalloc(&d_lut, (pfb ? 4 : 1) * num_packets * sizeof(uint64_t)));
    const __half* d_proto = nullptr;
    if (pfb) {
        // The hamming windowed-sinc prototype rounded to __half, its symmetric first half
        // (2 * fft_length values), one table per GPU shared across F-engine instances
        // (pfb_proto_shared).
        bool reused = false;
        d_proto = pfb_proto_shared(gpu_id, fft_length, 4, reused);
        if (reused)
            INFO("Reusing the shared 4-tap PFB prototype table on GPU {:d}", gpu_id);
    }
    float* d_gains = nullptr;
    float* d_stats = nullptr;
    if (gain_buf)
        CHECK_CUDA_ERROR(cudaMalloc(&d_gains, m * sizeof(float)));
    if (stats_buf) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_stats, m * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_stats, 0, m * sizeof(float)));
    }
    // The fused path quantizes in transform-local order: bytes staged for the transpose,
    // and the gain/stats tables reordered to match.
    uint8_t* d_q = nullptr;
    float* d_gains_t = nullptr;
    float* d_stats_t = nullptr;
    if (fused) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_q, m));
        if (gain_buf)
            CHECK_CUDA_ERROR(cudaMalloc(&d_gains_t, m * sizeof(float)));
        if (stats_buf) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_stats_t, m * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_stats_t, 0, m * sizeof(float)));
        }
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
    INFO("FFT of {:d} points ({:d} packets) on GPU {:d}, {:s}{:s}{:s}, scale {:g}", fft_length,
         num_packets, gpu_id, pfb ? "4-tap PFB, " : "",
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
        std::vector<uint64_t> warm_lut((pfb ? 4 : 1) * num_packets, (uint64_t)(uintptr_t)scratch);
        CHECK_CUDA_ERROR(cudaMemcpy(d_lut, warm_lut.data(), warm_lut.size() * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice));
        if (fourstep && fp16) {
            if (pfb) {
                pfb_taps taps;
                for (int q = 0; q < 4; q++)
                    taps.lut[q] = d_lut + q * num_packets;
                gpu_pfb_unpack12_fp16_t(taps, samples_per_packet, m1, m2, d_proto, d_a, stream);
            } else
                gpu_unpack12_fp16_t(d_lut, samples_per_packet, m1, m2, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m1, m2, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan, d_a, d_b, CUFFT_FORWARD));
            gpu_twiddle_transpose(d_b, m1, m2, d_whi, d_wlo, 1.0f / m1, split1, d_a, stream);
            if (fused) {
                gpu_tensor_fft2_quantize(d_a, m1, m2, d_whi, d_wlo, scale, gain, nullptr, nullptr,
                                         d_q, stream);
                gpu_bytes_transpose(d_q, m1, m2, (uint8_t*)d_b, stream);
            } else if (tensor) {
                gpu_tensor_fft(d_a, d_b, m2, m1, stream);
            } else {
                CHECK_CUFFT(cufftXtExec(plan2, d_a, d_b, CUFFT_FORWARD));
            }
            if (!fused)
                gpu_quantize_fourstep(d_b, m1, m2, d_whi, d_wlo, scale, gain, nullptr, nullptr,
                                      split2, (uint8_t*)d_a, stream);
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
    // PFB state: the rotating d_lut slot the next window lands in, how many previous
    // windows are valid history, and the release cursors of the last three jobs.
    uint32_t lut_slot = 0, history = 0;
    uint64_t rel_hist[3] = {0, 0, 0};
    size_t rel_idx = 0;

    // Hold processing until the first gain table, so no window goes out with the wrong gain.
    if (gain_buf) {
        const uint8_t* table = gain_buf->wait_for_full_frame(unique_name, gain_id);
        if (table == nullptr)
            return;
        CHECK_CUDA_ERROR(cudaMemcpy(d_gains, table, m * sizeof(float), cudaMemcpyHostToDevice));
        if (fused) {
            gpu_reorder_f32(d_gains, m1, m2, d_gains_t, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }
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
        // After a (re)alignment the previous windows are not contiguous with this one.
        if (pfb && (job->flags & FFT_WINDOW_REALIGNED))
            history = 0;

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
                if (fused)
                    gpu_reorder_f32(d_gains, m1, m2, d_gains_t, stream);
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
        uint64_t* d_job_lut = d_lut + (pfb ? lut_slot * num_packets : 0);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_job_lut, fft_window_lut(job),
                                         num_packets * sizeof(uint64_t), cudaMemcpyHostToDevice,
                                         stream));
        if (fourstep && fp16) {
            if (pfb) {
                // Slot lut_slot is the newest window; a null table folds zeros for history
                // the aligner has not produced since the last realignment.
                pfb_taps taps;
                for (uint32_t q = 0; q < 4; q++)
                    taps.lut[q] =
                        3 - q <= history ? d_lut + ((lut_slot + 1 + q) % 4) * num_packets : nullptr;
                gpu_pfb_unpack12_fp16_t(taps, samples_per_packet, m1, m2, d_proto, d_a, stream);
            } else
                gpu_unpack12_fp16_t(d_lut, samples_per_packet, m1, m2, d_a, stream);
            if (tensor)
                gpu_tensor_fft(d_a, d_b, m1, m2, stream);
            else
                CHECK_CUFFT(cufftXtExec(plan, d_a, d_b, CUFFT_FORWARD));
            gpu_twiddle_transpose(d_b, m1, m2, d_whi, d_wlo, 1.0f / m1, split1, d_a, stream);
            if (fused) {
                gpu_tensor_fft2_quantize(d_a, m1, m2, d_whi, d_wlo, scale, gain, d_gains_t,
                                         d_stats_t, d_q, stream);
                gpu_bytes_transpose(d_q, m1, m2, out_frame, stream);
            } else if (tensor) {
                gpu_tensor_fft(d_a, d_b, m2, m1, stream);
            } else {
                CHECK_CUFFT(cufftXtExec(plan2, d_a, d_b, CUFFT_FORWARD));
            }
            if (!fused)
                gpu_quantize_fourstep(d_b, m1, m2, d_whi, d_wlo, scale, gain, d_gains, d_stats,
                                      split2, out_frame, stream);
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
            if (fused) {
                gpu_reorder_f32(d_stats_t, m2, m1, d_stats, stream);
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_stats_t, 0, m * sizeof(float), stream));
            }
            CHECK_CUDA_ERROR(cudaMemcpyAsync(stats_frame, d_stats, m * sizeof(float),
                                             cudaMemcpyDeviceToHost, stream));
            if (!fused)
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_stats, 0, m * sizeof(float), stream));
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        gpu_seconds_gauge.set(current_time() - start);
        window_count++;
        if (emit_stats) {
            stats_buf->mark_frame_full(unique_name, stats_id);
            stats_id++;
        }

        // The PFB reads three windows of history: hold this job's frames and release what
        // job s-3 released.
        uint64_t release_target = job->release_up_to;
        if (pfb) {
            release_target = rel_hist[rel_idx];
            rel_hist[rel_idx] = job->release_up_to;
            rel_idx = (rel_idx + 1) % 3;
            lut_slot = (lut_slot + 1) % 4;
            if (history < 3)
                history++;
        }
        while (released < release_target) {
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
    // d_proto stays: the shared prototype table outlives any one instance.
    if (d_gains)
        CHECK_CUDA_ERROR(cudaFree(d_gains));
    if (d_stats)
        CHECK_CUDA_ERROR(cudaFree(d_stats));
    if (d_q)
        CHECK_CUDA_ERROR(cudaFree(d_q));
    if (d_gains_t)
        CHECK_CUDA_ERROR(cudaFree(d_gains_t));
    if (d_stats_t)
        CHECK_CUDA_ERROR(cudaFree(d_stats_t));
    CHECK_CUDA_ERROR(cudaFree(d_whi));
    CHECK_CUDA_ERROR(cudaFree(d_wlo));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
