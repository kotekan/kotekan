#include "fftwEngine.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(fftwEngine);

fftwEngine::fftwEngine(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fftwEngine::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 1024);

    samples = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * spectrum_length);
    spectrum = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * spectrum_length);
    fft_plan =
        (fftwf_plan_s*)fftwf_plan_dft_1d(spectrum_length, samples, spectrum, -1, FFTW_ESTIMATE);
}

fftwEngine::~fftwEngine() {
    fftwf_free(samples);
    fftwf_free(spectrum);
    fftwf_free(fft_plan);
}

void fftwEngine::main_thread() {
    int16_t* in_local;
    fftwf_complex* out_local;

    frame_in = 0;
    frame_out = 0;

    int BYTES_PER_SAMPLE = 2;
    int samples_per_input_frame = in_buf->frame_size / BYTES_PER_SAMPLE;

    while (!stop_thread) {
        in_local = (short*)wait_for_full_frame(in_buf, unique_name.c_str(), frame_in);
        if (in_local == nullptr)
            break;
        out_local = (fftwf_complex*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_out);
        if (out_local == nullptr)
            break;

        for (int j = 0; j < samples_per_input_frame / 2; j += spectrum_length) {
            DEBUG("Running FFT, {:i}", in_local[2 * j]);
            for (int i = 0; i < spectrum_length; i++) {
                samples[i][0] = in_local[2 * (i + j)];
                samples[i][1] = in_local[2 * (i + j) + 1];
            }
            fftwf_execute(fft_plan);
            memcpy(out_local, spectrum + spectrum_length / 2,
                   sizeof(fftwf_complex) * spectrum_length / 2);
            memcpy(out_local + spectrum_length / 2, spectrum,
                   sizeof(fftwf_complex) * spectrum_length / 2);
            out_local += spectrum_length;
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_in);
        mark_frame_full(out_buf, unique_name.c_str(), frame_out);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        frame_out = (frame_out + 1) % out_buf->num_frames;
    }
}
