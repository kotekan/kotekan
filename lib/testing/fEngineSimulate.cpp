#include "fEngineSimulate.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for int4x2_swapped_withoffset_t
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "crsPacket.hpp"       // for unpack12
#include "fftWindowJob.hpp"    // for fft_window_job, fft_window_lut, fft_window_job_size
#include "fftwPlannerLock.hpp" // for fftw_planner_mutex
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "visUtil.hpp"         // for frameID

#include "fmt.hpp" // for format

#include <algorithm>  // for clamp
#include <functional> // for bind
#include <math.h>     // for lrintf
#include <mutex>      // for mutex, lock_guard
#include <stdexcept>  // for invalid_argument

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(fEngineSimulate);

fEngineSimulate::fEngineSimulate(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fEngineSimulate::main_thread, this)),
    job_buf(get_buffer("job_buf")), payload_buf(get_buffer("payload_buf")),
    noise_buf(get_buffer("noise_buf")), out_buf(get_buffer("out_payload_buf")) {

    fft_length = config.get<uint32_t>(unique_name, "fft_length");
    samples_per_packet = config.get_default<uint32_t>(unique_name, "samples_per_packet", 4096);
    gain = config.get_default<float>(unique_name, "gain", 1.0f);
    const uint32_t bins_per_packet =
        config.get_default<uint32_t>(unique_name, "bins_per_packet", 8192);

    if (fft_length == 0 || fft_length % samples_per_packet != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: fft_length {:d} is not a multiple of samples_per_packet {:d}"),
                        unique_name, fft_length, samples_per_packet));
    num_packets = fft_length / samples_per_packet;
    num_bins = fft_length / 2;
    if (job_buf->frame_size < fft_window_job_size(num_packets))
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: job_buf frames hold {:d} bytes but a window of {:d} packets needs {:d}"),
            unique_name, job_buf->frame_size, num_packets, fft_window_job_size(num_packets)));
    if (out_buf->frame_size != num_bins)
        throw std::invalid_argument(fmt::format(
            fmt("{:s}: out_payload_buf frames hold {:d} bytes but a spectrum has {:d} bins"),
            unique_name, out_buf->frame_size, num_bins));
    if (bins_per_packet == 0 || num_bins % bins_per_packet != 0)
        throw std::invalid_argument(
            fmt::format(fmt("{:s}: bins_per_packet {:d} does not divide fft_length/2 = {:d}"),
                        unique_name, bins_per_packet, num_bins));

    job_buf->register_consumer(unique_name);
    payload_buf->register_consumer(unique_name);
    noise_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

fEngineSimulate::~fEngineSimulate() {
    if (plan) {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fftwf_destroy_plan(plan);
    }
    if (samples)
        fftwf_free(samples);
    if (spectrum)
        fftwf_free(spectrum);
}

void fEngineSimulate::main_thread() {
    // The lookup tables point into the noise frame; take it once and hold it forever.
    if (noise_buf->wait_for_full_frame(unique_name, 0) == nullptr)
        return;

    // fftwf_malloc is thread-safe; only the planner call needs the lock (fftwPlannerLock.hpp).
    samples = (float*)fftwf_malloc(sizeof(float) * fft_length);
    spectrum = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (num_bins + 1));
    {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        plan = fftwf_plan_dft_r2c_1d((int)fft_length, samples, spectrum, FFTW_ESTIMATE);
    }

    frameID job_id(job_buf), out_id(out_buf);
    frameID acquire_id(payload_buf), release_id(payload_buf);
    uint64_t acquired = 0, released = 0;
    const float scale = 2.0f / (float)fft_length;

    while (!stop_thread) {
        const uint8_t* job_frame = job_buf->wait_for_full_frame(unique_name, job_id);
        if (job_frame == nullptr)
            break;
        const fft_window_job* job = reinterpret_cast<const fft_window_job*>(job_frame);
        const uint64_t* lut = fft_window_lut(job);
        if (job->num_packets != num_packets)
            FATAL_ERROR("Job holds {:d} packets but fft_length/samples_per_packet is {:d}",
                        job->num_packets, num_packets);

        // Hold every payload frame up to the job's cursor, frame by frame (fftWindowJob.hpp).
        while (acquired < job->acquire_up_to) {
            if (payload_buf->wait_for_full_frame(unique_name, acquire_id) == nullptr)
                return;
            acquire_id++;
            acquired++;
        }

        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out_frame == nullptr)
            break;

        for (uint32_t i = 0; i < num_packets; i++) {
            const uint8_t* slot = reinterpret_cast<const uint8_t*>((uintptr_t)lut[i]);
            float* dst = samples + (size_t)i * samples_per_packet;
            for (uint32_t k = 0; k < samples_per_packet; k++)
                dst[k] = unpack12(slot, k) / 2048.0f;
        }
        fftwf_execute(plan);

        // Bins 0..fft_length/2-1: DC kept, Nyquist dropped. lrintf rounds half to even.
        for (uint32_t k = 0; k < num_bins; k++) {
            const long re = std::clamp(lrintf(gain * scale * spectrum[k][0]), -7l, 7l);
            const long im = std::clamp(lrintf(gain * scale * spectrum[k][1]), -7l, 7l);
            // Constructed as (imag, real): the second value lands in the high nibble.
            out_frame[k] = kotekan::int4x2_swapped_withoffset_t((int8_t)im, (int8_t)re).val;
        }

        while (released < job->release_up_to) {
            payload_buf->mark_frame_empty(unique_name, release_id);
            release_id++;
            released++;
        }

        job_buf->mark_frame_empty(unique_name, job_id);
        job_id++;
        out_buf->mark_frame_full(unique_name, out_id);
        out_id++;
    }
}
