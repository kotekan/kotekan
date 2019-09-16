#include <string>

using std::string;

#include "chimeMetadata.h"
#include "integrateHFBData.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(integrateHFBData);

integrateHFBData::integrateHFBData(Config& config_, const string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&integrateHFBData::main_thread, this)) {

    // Apply config.
    _num_frb_total_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    _num_frames_to_integrate =
        config.get_default<uint32_t>(unique_name, "num_frames_to_integrate", 80);
    _num_sub_freqs = config.get<uint32_t>(unique_name, "num_sub_freqs");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _good_samples_threshold = config.get<float>(unique_name, "good_samples_threshold");

    in_buf = get_buffer("hfb_input_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("hfb_output_buf");
    register_producer(out_buf, unique_name.c_str());
}

integrateHFBData::~integrateHFBData() {}

void integrateHFBData::initFirstFrame(float* input_data, float* sum_data,
                                      const uint32_t in_buffer_ID) {

    int64_t fpga_seq_num_start =
        fpga_seq_num_end - (_num_frames_to_integrate - 1) * _samples_per_data_set;
    memcpy(&sum_data[0], &input_data[0], _num_frb_total_beams * _num_sub_freqs * sizeof(float));
    total_lost_timesamples += get_fpga_seq_num(in_buf, in_buffer_ID) - fpga_seq_num_start;
    // Get the first FPGA sequence no. to check for missing frames
    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);
    frame++;
}

void integrateHFBData::integrateFrame(float* input_data, float* sum_data,
                                      const uint32_t in_buffer_ID) {
    fpga_seq_num += _samples_per_data_set;
    total_lost_timesamples += get_fpga_seq_num(in_buf, in_buffer_ID) - fpga_seq_num;

    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);

    DEBUG("\nIntegrate frame {:d}, total_lost_timesamples: {:d}...\n", frame + 1,
          total_lost_timesamples);

    // Integrates data from the input buffer to the output buffer.
    for (uint beam = 0; beam < _num_frb_total_beams; beam++) {
        for (uint freq = 0; freq < _num_sub_freqs; freq++) {
            sum_data[beam * _num_sub_freqs + freq] += input_data[beam * _num_sub_freqs + freq];
        }
    }
}

void integrateHFBData::normaliseFrame(float* sum_data, const uint32_t in_buffer_ID) {

    const float normalise_frac =
        (float)total_timesamples / (total_timesamples - total_lost_timesamples);

    for (uint beam = 0; beam < _num_frb_total_beams; beam++) {
        for (uint freq = 0; freq < _num_sub_freqs; freq++) {
            sum_data[beam * _num_sub_freqs + freq] *= normalise_frac;
        }
    }

    DEBUG("Integration completed with {:d} lost samples", total_lost_timesamples);

    total_lost_timesamples = 0;
    frame = 0;

    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);
}

void integrateHFBData::main_thread() {

    uint in_buffer_ID = 0; // Process only 1 GPU buffer, cycle through buffer depth
    uint8_t* in_frame;
    int out_buffer_ID = 0;
    total_timesamples = _samples_per_data_set * _num_frames_to_integrate;
    total_lost_timesamples = 0;
    fpga_seq_num = 0, fpga_seq_num_end = (_num_frames_to_integrate - 1) * _samples_per_data_set;
    frame = 0;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == NULL)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == NULL)
            goto end_loop;

        float* sum_data = (float*)out_buf->frames[out_buffer_ID];
        float* input_data = (float*)in_buf->frames[in_buffer_ID];

        // Get the no. of lost samples in this frame
        total_lost_timesamples += get_lost_timesamples(in_buf, in_buffer_ID);

        // If we are on the first frame copy it directly into the
        // output buffer frame so that we don't need to zero the frame
        if (frame == 0
            && get_fpga_seq_num(in_buf, in_buffer_ID)
                       % (_num_frames_to_integrate * _samples_per_data_set)
                   == 0) {
            initFirstFrame(input_data, sum_data, in_buffer_ID);

            fpga_seq_num_end =
                fpga_seq_num + (_num_frames_to_integrate - 1) * _samples_per_data_set;
        } else {

            // TODO:JSW Store the amount of renormalisation used in the frame
            // Increment the no. of lost frames if there are missing frames
            // When all frames have been integrated output the result
            if (get_fpga_seq_num(in_buf, in_buffer_ID)
                >= fpga_seq_num_end + _samples_per_data_set) {

                total_lost_timesamples += fpga_seq_num_end - fpga_seq_num;

                const float good_samples_frac =
                    (float)(total_timesamples - total_lost_timesamples) / total_timesamples;

                // Normalise data
                normaliseFrame(sum_data, in_buffer_ID);

                // Only output integration if there are enough good samples
                if (good_samples_frac >= _good_samples_threshold) {
                    mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);

                    // Get a new output buffer
                    out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;
                    out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
                    if (out_frame == NULL)
                        goto end_loop;

                    sum_data = (float*)out_buf->frames[out_buffer_ID];
                }

                // Already started next integration
                if (fpga_seq_num > fpga_seq_num_end) {
                    fpga_seq_num_end =
                        fpga_seq_num_end + _num_frames_to_integrate * _samples_per_data_set;
                    initFirstFrame(input_data, sum_data, in_buffer_ID);
                }

            } else {

                if (frame == 0)
                    initFirstFrame(input_data, sum_data, in_buffer_ID);
                else {
                    integrateFrame(input_data, sum_data, in_buffer_ID);
                    frame++;
                }

                // When all frames have been integrated output the result
                if (get_fpga_seq_num(in_buf, in_buffer_ID) >= fpga_seq_num_end) {

                    total_lost_timesamples += fpga_seq_num_end - fpga_seq_num;

                    const float good_samples_frac =
                        (float)(total_timesamples - total_lost_timesamples) / total_timesamples;

                    // Normalise data
                    normaliseFrame(sum_data, in_buffer_ID);

                    fpga_seq_num_end =
                        fpga_seq_num_end + _num_frames_to_integrate * _samples_per_data_set;

                    // Only output integration if there are enough good samples
                    if (good_samples_frac >= _good_samples_threshold) {
                        mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);

                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;
                        out_frame =
                            wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
                        if (out_frame == NULL)
                            goto end_loop;

                        sum_data = (float*)out_buf->frames[out_buffer_ID];
                    }
                }
            }
        }

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;

    } // end stop thread
end_loop:;
}
