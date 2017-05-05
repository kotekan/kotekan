#define __STDC_FORMAT_MACROS 1
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <functional>

#include "buffers.h"
#include "errors.h"
#include "output_formating.h"
#include "Config.hpp"
#include "gpuPostProcess.hpp"
#include "util.h"
#include "version.h"

gpuPostProcess::gpuPostProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config_, unique_name, buffer_container,
                       std::bind(&gpuPostProcess::main_thread, this)) {
    apply_config(0);

    out_buf = get_buffer("chrx_out_buf");
    gate_buf = get_buffer("gate_out_buf");

    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof(struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("corr_in_buf_" + std::to_string(i));
    }
}

gpuPostProcess::~gpuPostProcess() {
    if (_product_remap_c != nullptr)
        delete _product_remap_c;

    if (_rest_copy_vis != nullptr)
        delete _rest_copy_vis;

    free(in_buf);
}

void gpuPostProcess::vis_endpoint(connectionInstance& conn, json& json_request) {
    //std::lock_guard<std::mutex> lock(_rest_copy_lock);
    int freq = 0;
    try {
        freq = json_request["freq"];
    } catch (...) {
        conn.send_error("Could not parse freq", STATUS_BAD_REQUEST);
    }
    if (freq > _num_total_freq) {
        conn.send_error("freq out of range", STATUS_BAD_REQUEST);
    }
    int num_values = ((_num_elem * (_num_elem + 1)) / 2 );
    conn.send_binary_reply((uint8_t *)&_rest_copy_vis[num_values * freq], num_values * sizeof(complex_int_t));
}

void gpuPostProcess::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _num_elem = config.get_int(unique_name, "num_elements");
    _num_total_freq = config.get_int(unique_name, "num_total_freq");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
    _num_blocks = config.get_int(unique_name, "num_blocks");
    _block_size = config.get_int(unique_name, "block_size");
    _link_map = config.get_int_array(unique_name, "link_map");
    _num_fpga_links = config.get_int(unique_name, "num_links");
    _enable_basic_gating = config.get_bool(unique_name, "enable_gating");
    _gate_phase = config.get_int(unique_name, "gate_phase");
    _gate_cadence = config.get_int(unique_name, "gate_cadence");
    _num_gpus = config.get_int(unique_name, "num_gpus");
    _product_remap = config.get_int_array(unique_name, "product_remap");

    // Create a C style array for backwards compatiably.
    if (_product_remap_c != NULL)
        delete _product_remap_c;

    _product_remap_c = new int32_t[_product_remap.size()];
    for (uint32_t i = 0; i < _product_remap.size(); ++i) {
        _product_remap_c[i] = _product_remap[i];
    }
}

// TODO, this needs a re-factor to reduce complexity.
void gpuPostProcess::main_thread() {

    apply_config(0);

    int in_buffer_ID = -1;
    int out_buffer_ID = 0;
    int frame_number = 0;

    int useableBufferIDs[_num_gpus][1];
    for (int i = 0; i < _num_gpus; ++i) {
        useableBufferIDs[i][0] = 0;
    }
    int link_id = 0;

    // Create tcp send buffer
    const int num_values = ((_num_elem * (_num_elem + 1)) / 2 ) * _num_total_freq;
    const int buffer_size = sizeof(struct tcp_frame_header) +
        num_values * sizeof(complex_int_t) +
        _num_total_freq * sizeof(struct per_frequency_data) +
        _num_total_freq * _num_elem * sizeof(struct per_element_data) +
        num_values * sizeof(uint8_t);

    assert(buffer_size == out_buf->buffer_size);

    const int num_vis = ((_num_elem * (_num_elem + 1)) / 2 );
    const int num_values_per_link = num_vis * _num_local_freq;

    unsigned char * buf = (unsigned char *)malloc(buffer_size);
    CHECK_MEM(buf);

    unsigned char * data_sets_buf =
        (unsigned char *)malloc(num_values * _num_data_sets * sizeof(complex_int_t));
    CHECK_MEM(data_sets_buf);

    struct per_frequency_data ** local_freq_data = (struct per_frequency_data **)malloc(_num_data_sets * sizeof(void *));
    CHECK_MEM(local_freq_data);

    struct per_element_data ** local_element_data = (struct per_element_data **)malloc(_num_data_sets * sizeof(void *));
    for (int i = 0; i < _num_data_sets; ++i) {
        local_freq_data[i] = (struct per_frequency_data*)malloc(_num_total_freq * sizeof(struct per_frequency_data));
        CHECK_MEM(local_freq_data[i]);

        local_element_data[i] = (struct per_element_data*)malloc(_num_total_freq * _num_elem *
                                    sizeof(struct per_element_data));
        CHECK_MEM(local_element_data[i]);
    }

    // Create convenient pointers into the buffer (yay pointer math).
    struct tcp_frame_header * header = (struct tcp_frame_header *)buf;

    int offset = sizeof(struct tcp_frame_header);
    complex_int_t * visibilities = ( complex_int_t * )&buf[offset];

    offset += num_values * sizeof(complex_int_t);
    struct per_frequency_data * frequency_data = ( struct per_frequency_data * )&buf[offset];

    offset += _num_total_freq * sizeof(struct per_frequency_data);
    struct per_element_data * element_data = (struct per_element_data *)&buf[offset];

    offset += _num_total_freq * _num_elem *
              sizeof(struct per_element_data);
    uint8_t * vis_weight = (uint8_t *)&buf[offset];

    // Safety check for pointer math.
    offset += num_values * sizeof(uint8_t);
    assert(offset == buffer_size);

    // Add version information to the header.
    strcpy(header->kotekan_git_hash, GIT_COMMIT_HASH);
    header->kotekan_version = KOTEKAN_VERSION;

    // This is a bit of a hack for gating, there is are better ways to do this.
    int gated_buf_size = sizeof(struct gate_frame_header)
                                + num_values * sizeof(complex_int_t);
    unsigned char * gated_buf = (unsigned char *) malloc(gated_buf_size);
    CHECK_MEM(gated_buf);

    struct gate_frame_header * gate_header = (struct gate_frame_header *)gated_buf;
    complex_int_t * gated_vis = (complex_int_t *)(gated_buf + sizeof(struct gate_frame_header));

    // Changing destination pointer for the different gates
    complex_int_t * vis = visibilities;

    // REST server section
    // TODO move to a function?
    // TODO also clean up this function.
    _rest_copy_vis = new complex_int_t[num_values];
    using namespace std::placeholders;
    restServer * rest_server = get_rest_server();
    rest_server->register_json_callback("/vis", std::bind(&gpuPostProcess::vis_endpoint, this, _1, _2));
    // end REST server section

    // Wait for full buffers.
    while(!stop_thread) {

        int gpu_id = _link_map[link_id];

        // This call is blocking!
        in_buffer_ID = get_full_buffer_from_list(in_buf[gpu_id], useableBufferIDs[gpu_id], 1);
        INFO("GPU Post process got full buffer ID %d for GPU %d", useableBufferIDs[gpu_id][0], gpu_id);

        // Check if the producer has finished, and we should exit.
        if (in_buffer_ID == -1) {
            mark_producer_done(out_buf, 0);
            INFO("Closing gpu_post_process");
            int ret;
            pthread_exit((void *) &ret);
        }

        // TODO Check that this is valid.  Make sure all seq numbers are the same for a frame, etc.
        uint64_t fpga_seq_number = get_fpga_seq_num(in_buf[gpu_id], in_buffer_ID);
        struct timeval frame_start_time = get_first_packet_recv_time(in_buf[gpu_id], in_buffer_ID);

        for (int i = 0; i < _num_data_sets; ++i) {

            if (_num_elem <= 16) {
                // TODO Make this cleaner (single function)
                reorganize_32_to_16_element_GPU_correlated_data_with_shuffle(
                    _num_local_freq,
                    _num_elem,
                    1,
                    (int *)&in_buf[gpu_id]->data[in_buffer_ID][i * (in_buf[gpu_id]->buffer_size / _num_data_sets)],
                    _product_remap_c);


                full_16_element_matrix_to_upper_triangle(
                    _num_local_freq,
                    (int *)&in_buf[gpu_id]->data[in_buffer_ID][i * (in_buf[gpu_id]->buffer_size / _num_data_sets)],
                    (complex_int_t *)&data_sets_buf[(i * num_values + link_id * num_values_per_link) * sizeof(complex_int_t)]);
            } else {
                reorganize_GPU_to_upper_triangle_remap(_block_size,
                    _num_blocks,
                    _num_local_freq,
                    _num_elem,
                    1,
                    (int *)&in_buf[gpu_id]->data[in_buffer_ID][i * (in_buf[gpu_id]->buffer_size / _num_data_sets)],
                    (complex_int_t *)&data_sets_buf[(i * num_values + link_id * num_values_per_link) * sizeof(complex_int_t)],
                    _product_remap_c);
            }

            // Frequency varying data.
            uint32_t packed_stream_ID = get_streamID(in_buf[gpu_id], in_buffer_ID);
            struct ErrorMatrix * error_matrix = get_error_matrix(in_buf[gpu_id], in_buffer_ID);
            for (int j = 0; j < _num_local_freq; ++j) {
                int pos = link_id*_num_local_freq + j;
                local_freq_data[i][pos].stream_id.link_id = packed_stream_ID & 0x000F;
                local_freq_data[i][pos].stream_id.slot_id = (packed_stream_ID & 0x00F0) >> 4;
                local_freq_data[i][pos].stream_id.crate_id = (packed_stream_ID & 0x0F00) >> 8;
                local_freq_data[i][pos].stream_id.reserved = (packed_stream_ID & 0xF000) >> 12;
                local_freq_data[i][pos].index = j;
                // TODO this needs to be data set aware.  adjust the error matrix code for this.
                local_freq_data[i][pos].lost_packet_count = error_matrix->bad_timesamples;
                local_freq_data[i][pos].rfi_count = 0;  // TODO add RFI counts here.

                // Frequency and element varing data.
                for (int e = 0; e < _num_elem; ++e) {
                    pos = link_id * _num_elem * _num_local_freq +
                        j * _num_elem + _product_remap[e];
                    // TODO Set these values with the error matrix.
                    local_element_data[i][pos].fpga_adc_count = 0;
                    local_element_data[i][pos].fpga_fft_count = 0;
                    local_element_data[i][pos].fpga_scalar_count = 0;
                }
            }
        }

        // Only happens once every time all the links have been read from.
        if (link_id + 1 == _num_fpga_links) {

            // Gating data.
            // Phase = 0 means the noise source ON bin starts at 0
            if (_enable_basic_gating == 1) {
                int64_t intergration_num = fpga_seq_number / _samples_per_data_set;

                int64_t step = (intergration_num / _gate_cadence) + _gate_phase;

                if (step % 2 == 0) {
                    vis = gated_vis;
                } else {
                    vis = visibilities;
                }
            }

            // Happens once for each data set within the frames.
            for (int i = 0; i < _num_data_sets; ++i) {

                // If this is the first frame, set the header, and initial visibility data.
                if (frame_number == 0) {
                    header->cpu_timestamp = frame_start_time;
                    double time_offset = i * (_samples_per_data_set * 2.56);
                    header->cpu_timestamp.tv_usec += time_offset;
                    header->fpga_seq_number = fpga_seq_number + i * _samples_per_data_set;
                    header->num_freq = _num_total_freq;
                    header->num_vis = num_vis;
                    header->num_elements = _num_elem;
                    header->num_links = _num_fpga_links;

                    if (_enable_basic_gating) {
                        snprintf(gate_header->description, MAX_GATE_DESCRIPTION_LEN, "ON - OFF");
                        gate_header->folding_period = (double)_gate_cadence *
                                            2.56 * (double)_samples_per_data_set;
                        gate_header->folding_start = (double)frame_start_time.tv_sec * 1000.0 * 1000.0 +
                                                     (double)frame_start_time.tv_usec;
                        // Convert to seconds
                        gate_header->folding_period /= 1000000.0;
                        gate_header->folding_start /= 1000000.0;
                        gate_header->fpga_count_start = fpga_seq_number;
                        gate_header->set_num = 1; // TODO This shouldn't be hard coded!!
                        gate_header->gate_weight[0] = (_gate_phase == 0) ? 1.0 : -1.0;
                        gate_header->gate_weight[1] = (_gate_phase == 0) ? -1.0 : 1.0;

                        header->num_gates = 1;
                    }

                    for (int j = 0; j < num_values; ++j) {
                        vis[j] = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                        vis_weight[j] = 0xFF;  // TODO Set this with the error matrix
                    }
                    for (int j = 0; j < _num_total_freq; ++j) {
                        frequency_data[j] = local_freq_data[i][j];
                    }
                    for (int j = 0; j < _num_elem * _num_total_freq; ++j) {
                        element_data[j] = local_element_data[i][j];
                    }

                } else if (frame_number == _gate_cadence) {
                    // This will either be start of the ON data or the first frame of OFF data
                    // so we need to make sure we reset the values here.
                    for (int j = 0; j < num_values; ++j) {
                        vis[j] = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                        vis_weight[j] = 0xFF;  // TODO Set this with the error matrix
                    }
                } else {
                    // Add to the visibilities.
                    for (int j = 0; j < num_values; ++j) {
                        complex_int_t temp_vis = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                        vis[j].real += temp_vis.real;
                        vis[j].imag += temp_vis.imag;
                    }
                    for (int j = 0; j < _num_total_freq; ++j) {
                      frequency_data[j].lost_packet_count += local_freq_data[i][j].lost_packet_count;
                      frequency_data[j].rfi_count += local_freq_data[i][j].rfi_count;
                    }
                }

                // If we are on the last frame in the set, push the buffer to the network thread.
                if (frame_number + 1 >= _num_gpu_frames) {
                    //INFO("Sending frame to network thread: FPGA_SEQ_NUMBER = %u ; NUM_FREQ = %d ; NUM_VIS = %d ; BUFFER_SIZE = %d",
                    //     header->fpga_seq_number,
                    //     buffer_size);

                    char frame_loss_str[20 * _num_total_freq / _num_local_freq];
                    char tmp_str[20];
                    strcpy (frame_loss_str, " ");
                    for (int j = 0; j < _num_total_freq / _num_local_freq; ++j) {
                        snprintf(tmp_str, 20, "%.6f%%; ",
                                 (float)100 * (float)frequency_data[j * _num_local_freq].lost_packet_count /
                                 (float)(_samples_per_data_set * _num_gpu_frames));
                        strcat(frame_loss_str, tmp_str);
                    }
                    INFO("Frame %" PRIu64 " loss rates:%s", header->fpga_seq_number, frame_loss_str);

                    wait_for_empty_buffer(out_buf, out_buffer_ID);
                    wait_for_empty_buffer(gate_buf, out_buffer_ID);

                    if (_enable_basic_gating) {
                        DEBUG("Copying gated data to the gate_buf!");
                        for (int j = 0; j < num_values; ++j) {
                            // Visibilities = OFF + ON
                            // gated_vis = ON - OFF
                            gated_vis[j].real = gated_vis[j].real - visibilities[j].real;
                            gated_vis[j].imag = gated_vis[j].imag - visibilities[j].imag;
                            visibilities[j].real = gated_vis[j].real + 2*visibilities[j].real;
                            visibilities[j].imag = gated_vis[j].imag + 2*visibilities[j].imag;
                        }
                        memcpy(gate_buf->data[out_buffer_ID], gated_buf, gated_buf_size);
                        mark_buffer_full(gate_buf, out_buffer_ID);
                    }

                    // memcpy visibilities into REST buffer.
                    {
                        //std::lock_guard<std::mutex> lock(_rest_copy_lock);
                        memcpy(_rest_copy_vis, visibilities, num_values * sizeof(complex_int_t));
                    }

                    memcpy(out_buf->data[out_buffer_ID], buf, buffer_size);
                    mark_buffer_full(out_buf, out_buffer_ID);
                    INFO("gpu_post_process: marked output buffer full: %d", out_buffer_ID );

                    out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_buffers;
                }
            }

            frame_number = (frame_number + 1) % _num_gpu_frames;
        }

        release_info_object(in_buf[gpu_id], in_buffer_ID);
        mark_buffer_empty(in_buf[gpu_id], in_buffer_ID);
        INFO("gpu_post_process: marked in buffer empty: gpu_id %d, buffer id %d", gpu_id, in_buffer_ID );

        useableBufferIDs[gpu_id][0] = (useableBufferIDs[gpu_id][0] + 1) % in_buf[gpu_id]->num_buffers;

        link_id = (link_id + 1) % _num_fpga_links;
    }

}
