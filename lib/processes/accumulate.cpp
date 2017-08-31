#include "accumulate.hpp"
#include "errors.h"
#include "fpga_header_functions.h"

accumulate::accumulate(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&accumulate::main_thread, this)) {

    apply_config(0);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

accumulate::~accumulate() {
}

void accumulate::apply_config(uint64_t fpga_seq) {
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
}

void accumulate::main_thread() {

    int in_buf_id = 0;
    int out_buf_id = 0;
    int64_t frame_id = 0;
    int32_t * input;
    int32_t * output;
    struct rawGPUFrameHeader * header;
    uint64_t seq_num;

    for (;;) {
        wait_for_full_buffer(in_buf, unique_name.c_str(), in_buf_id);
        input = (int32_t *)in_buf->data[in_buf_id];

        seq_num = get_fpga_seq_num(in_buf, in_buf_id);

        if (frame_id % _num_gpu_frames == 0) {
            wait_for_empty_buffer(out_buf, unique_name.c_str(), out_buf_id);
            header = (struct rawGPUFrameHeader *)out_buf->data[out_buf_id];
            output = (int32_t *)(&out_buf->data[out_buf_id][sizeof(struct rawGPUFrameHeader)]);

            header->fpga_seq_num = get_fpga_seq_num(in_buf, in_buf_id);
            uint16_t stream_id = get_streamID(in_buf, in_buf_id);
            stream_id_t s_stream_id = extract_stream_id(stream_id);
            s_stream_id.crate_id = 2; // TODO set this to match the number of crate-pairs active.
            header->stream_id = encode_stream_id(s_stream_id);
            timeval time_v = get_first_packet_recv_time(in_buf, in_buf_id);
            header->epoch_time_sec = (uint32_t)time_v.tv_sec;
            header->epoch_time_usec = (uint32_t)time_v.tv_usec;
            header->unused = 2048;

            struct ErrorMatrix * error_matrix = get_error_matrix(in_buf, in_buf_id);
            header->lost_frames = error_matrix->bad_timesamples;

            for (int i = 0; i < in_buf->buffer_size/sizeof(int32_t); ++i) {
                output[i] = input[i];
            }

        } else {
            struct ErrorMatrix * error_matrix = get_error_matrix(in_buf, in_buf_id);
            header->lost_frames += error_matrix->bad_timesamples;

            for (int i = 0; i < in_buf->buffer_size/sizeof(int32_t); ++i) {
                output[i] += input[i];
            }
        }

        release_info_object(in_buf, in_buf_id);
        mark_buffer_empty(in_buf, unique_name.c_str(), in_buf_id);
        in_buf_id = (in_buf_id + 1) % in_buf->num_buffers;
        frame_id++;

        if (frame_id % _num_gpu_frames == 0) {
            mark_buffer_full(out_buf, unique_name.c_str(), out_buf_id);
            out_buf_id = (out_buf_id + 1) % out_buf->num_buffers;
        }
    }
}
