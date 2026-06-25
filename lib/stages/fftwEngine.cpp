#include "fftwEngine.hpp"

#include <stdint.h>             // for int16_t
#include <string.h>             // for memcpy
#include <functional>           // for bind, function
#include <mutex>                // for mutex, lock_guard
#include <memory>               // for shared_ptr

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "airspyFrameDesc.hpp"  // for make_fengine_desc
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "fftwPlannerLock.hpp"  // for fftw_planner_mutex
#include "kotekanLogging.hpp"   // for DEBUG, FATAL_ERROR
#include "fmt.hpp"              // for compile_string_to_view
#include "NDArray.hpp"          // for GenericNDArray
#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan

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

    // Output is cfloat32 1-D regardless of input_type (we accept either
    // int16 reals or cint16 IQ pairs upstream, so in_buf's descriptor is
    // intentionally left for the upstream producer to assert).
    out_buf->set_frame_desc(
        kotekan_airspy::make_fengine_desc(out_buf->frame_size / sizeof(fftwf_complex)));

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

    // Optional polyphase filter bank: num_taps > 1 prepends a windowed-sinc
    // prototype + fold before each FFT; num_taps == 1 leaves the straight FFT
    // path untouched. fft_len is the transform size (2x in real mode).
    _num_taps = config.get_default<int>(unique_name, "num_taps", 1);
    if (_num_taps > 1) {
        const int fft_len = _real_input ? _spectrum_length * 2 : _spectrum_length;
        const std::string win =
            config.get_default<std::string>(unique_name, "pfb_window", "hamming");
        try {
            _proto = dsp::pfb_prototype(fft_len, _num_taps, dsp::window_from_string(win));
        } catch (const std::exception& e) {
            FATAL_ERROR("fftwEngine: {:s}", e.what());
            return;
        }
        if (_real_input) {
            _hist_r.assign(fft_len * _num_taps, 0.0f);
            _block_r.resize(fft_len);
        } else {
            _hist_c.assign(fft_len * _num_taps, std::complex<float>(0.0f, 0.0f));
            _block_c.resize(fft_len);
        }
    } else {
        _num_taps = 1;
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
    long long frames_produced = 0; // monotonic output-frame counter (absolute reference)

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
                if (_num_taps > 1) {
                    for (int i = 0; i < fft_len; i++)
                        _block_r[i] = (float)in_local[i + j] / _spectrum_length;
                    dsp::pfb_push(_hist_r.data(), _block_r.data(), fft_len, _num_taps);
                    dsp::pfb_fold(_hist_r.data(), _proto.data(), fft_len, _num_taps, real_samples);
                } else {
                    for (int i = 0; i < fft_len; i++) {
                        real_samples[i] = (float)in_local[i + j] / _spectrum_length;
                    }
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
                if (_num_taps > 1) {
                    for (int i = 0; i < _spectrum_length; i++)
                        _block_c[i] = std::complex<float>(in_local[2 * (i + j)],
                                                          in_local[2 * (i + j) + 1]);
                    dsp::pfb_push(_hist_c.data(), _block_c.data(), _spectrum_length, _num_taps);
                    dsp::pfb_fold(_hist_c.data(), _proto.data(), _spectrum_length, _num_taps,
                                  reinterpret_cast<std::complex<float>*>(complex_samples));
                } else {
                    for (int i = 0; i < _spectrum_length; i++) {
                        complex_samples[i][0] = in_local[2 * (i + j)];
                        complex_samples[i][1] = in_local[2 * (i + j) + 1];
                    }
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

        // Stamp the absolute sample reference (chordMetadata fpga_seq) so downstream
        // search/track -- possibly on other nodes after bufferSend/Recv -- share one
        // "sample 0" for code-phase referencing. Propagate the source seq if the input
        // carries it, else generate from the monotonic output-frame counter.
        if (out_buf->metadata_pool) {
            out_buf->allocate_new_metadata_object(frame_out);
            int64_t seq = frames_produced * (int64_t)samples_per_input_frame; // abs sample of hop 0
            if (metadata_is_gnss_chan(in_buf)) {
                auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
                if (mi && mi->fpga_seq >= 0) // propagate the source seq if present
                    seq = mi->fpga_seq;
            }
            get_gnss_chan_metadata(out_buf, frame_out)->fpga_seq = seq;
        }

        in_buf->mark_frame_empty(unique_name, frame_in);
        out_buf->mark_frame_full(unique_name, frame_out);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        frame_out = (frame_out + 1) % out_buf->num_frames;
        frames_produced++;
    }
}
