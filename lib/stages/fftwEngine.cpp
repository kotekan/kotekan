#include "fftwEngine.hpp"

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "fftwPlannerLock.hpp"  // for fftw_planner_mutex
#include "kotekanLogging.hpp"   // for DEBUG, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <functional> // for bind
#include <mutex>      // for lock_guard
#include <stdint.h>   // for int16_t
#include <string.h>   // for memcpy

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(fftwEngine);

fftwEngine::fftwEngine(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fftwEngine::main_thread, this)),
    real_samples(nullptr), complex_samples(nullptr), spectrum(nullptr), fft_plan(nullptr) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 128);

    std::string input_type = config.get_default<std::string>(unique_name, "input_type", "complex");
    if (input_type == "real") {
        _real_input = true;
    } else if (input_type == "complex") {
        _real_input = false;
    } else {
        FATAL_ERROR("fftwEngine: unknown input_type '{:s}'; expected 'complex' or 'real'.",
                    input_type);
        return;
    }

    // fftwf_malloc is thread-safe; only the planner call needs the lock.
    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    if (_real_input) {
        const int fft_len = _spectrum_length * 2;
        real_samples = (float*)fftwf_malloc(sizeof(float) * fft_len);
        spectrum = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (fft_len / 2 + 1));
        fft_plan = fftwf_plan_dft_r2c_1d(fft_len, real_samples, spectrum, FFTW_ESTIMATE);
    } else {
        complex_samples = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _spectrum_length);
        spectrum = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _spectrum_length);
        fft_plan =
            fftwf_plan_dft_1d(_spectrum_length, complex_samples, spectrum, -1, FFTW_ESTIMATE);
    }
}

fftwEngine::~fftwEngine() {
    if (fft_plan) {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fftwf_destroy_plan(fft_plan);
    }
    if (real_samples)
        fftwf_free(real_samples);
    if (complex_samples)
        fftwf_free(complex_samples);
    if (spectrum)
        fftwf_free(spectrum);
}

void fftwEngine::main_thread() {
    frame_in = 0;
    frame_out = 0;

    constexpr int BYTES_PER_SAMPLE = 2; // int16_t

    while (!stop_thread) {
        int16_t* in_local = (int16_t*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        fftwf_complex* out_local =
            (fftwf_complex*)out_buf->wait_for_empty_frame(unique_name, frame_out);
        if (out_local == nullptr)
            break;

        const int samples_per_input_frame = in_buf->frame_size / BYTES_PER_SAMPLE;

        if (_real_input) {
            const int fft_len = _spectrum_length * 2;
            for (int j = 0; j < samples_per_input_frame; j += fft_len) {
                DEBUG("Running real FFT, {:d}", in_local[j]);
                for (int i = 0; i < fft_len; i++) {
                    real_samples[i] = (float)in_local[i + j] / _spectrum_length;
                }
                fftwf_execute(fft_plan);
                // r2c gives fft_len/2+1 = _spectrum_length+1 bins; we drop Nyquist.
                memcpy(out_local, spectrum, sizeof(fftwf_complex) * _spectrum_length);
                out_local += _spectrum_length;
            }
        } else {
            // Complex IQ: each input sample is an int16 pair, so step by 2*_spectrum_length ints.
            for (int j = 0; j < samples_per_input_frame / 2; j += _spectrum_length) {
                DEBUG("Running complex FFT, {:d}", in_local[2 * j]);
                for (int i = 0; i < _spectrum_length; i++) {
                    complex_samples[i][0] = in_local[2 * (i + j)];
                    complex_samples[i][1] = in_local[2 * (i + j) + 1];
                }
                fftwf_execute(fft_plan);
                // Shift DC into the centre.
                memcpy(out_local, spectrum + _spectrum_length / 2,
                       sizeof(fftwf_complex) * _spectrum_length / 2);
                memcpy(out_local + _spectrum_length / 2, spectrum,
                       sizeof(fftwf_complex) * _spectrum_length / 2);
                out_local += _spectrum_length;
            }
        }

        in_buf->mark_frame_empty(unique_name, frame_in);
        out_buf->mark_frame_full(unique_name, frame_out);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        frame_out = (frame_out + 1) % out_buf->num_frames;
    }
}
