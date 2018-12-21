#include "simpleAutocorr.hpp"

REGISTER_KOTEKAN_PROCESS(simpleAutocorr);

simpleAutocorr::simpleAutocorr(Config& config, const string& unique_name,
                               bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&simpleAutocorr::main_thread, this)) {

    buf_in = get_buffer("in_buf");
    register_consumer(buf_in, unique_name.c_str());
    buf_out = get_buffer("out_buf");
    register_producer(buf_out, unique_name.c_str());

    spectrum_length = config.get_default<int>(unique_name, "spectrum_length", 1024);
    spectrum_out = (float*)calloc(spectrum_length, sizeof(float));
    integration_length = config.get_default<int>(unique_name, "integration_length", 1024);
}

simpleAutocorr::~simpleAutocorr() {
    free(spectrum_out);
}

void simpleAutocorr::main_thread() {
    float* in_local;
    uint* out_local;

    float re, im;
    frame_in = 0;
    frame_out = 0;
    int integration_ct = 0;
    int out_loc = 0;

    int samples_per_frame = buf_in->frame_size / (2 * sizeof(float));

    while (!stop_thread) {
        in_local = (float*)wait_for_full_frame(buf_in, unique_name.c_str(), frame_in);
        if (in_local == NULL)
            break;
        for (int j = 0; j < samples_per_frame; j += spectrum_length) {
            for (int i = 0; i < spectrum_length; i++) {
                re = in_local[(i + j) * 2];
                im = in_local[(i + j) * 2 + 1];
                spectrum_out[i] += (re * re + im * im) / integration_length;
            }
            integration_ct++;

            if (integration_ct >= integration_length) {
                if (out_loc == 0)
                    out_local =
                        (uint*)wait_for_empty_frame(buf_out, unique_name.c_str(), frame_out);
                for (int i = 0; i < spectrum_length; i++)
                    out_local[out_loc++] = spectrum_out[i];
                out_local[out_loc++] = integration_ct;

                if (out_loc * sizeof(uint) == (uint32_t)buf_out->frame_size) {
                    mark_frame_full(buf_out, unique_name.c_str(), frame_out);
                    frame_out = (frame_out + 1) % buf_out->num_frames;
                    out_loc = 0;
                    DEBUG("Finished integrating a frame!");
                }

                memset(spectrum_out, 0, spectrum_length * sizeof(float));
                integration_ct = 0;
            }
        }
        mark_frame_empty(buf_in, unique_name.c_str(), frame_in);
        frame_in = (frame_in + 1) % buf_in->num_frames;
    }
}
