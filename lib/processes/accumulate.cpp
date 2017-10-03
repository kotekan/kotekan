#include "accumulate.hpp"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

accumulate::accumulate(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&accumulate::main_thread, this)) {

    apply_config(0);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_gpu_frames = config.get_int(unique_name, "num_gpu_frames");
}

accumulate::~accumulate() {
}

void accumulate::apply_config(uint64_t fpga_seq) {
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
        uint8_t * in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_id);
        input = (int32_t *)in_frame;

        seq_num = get_fpga_seq_num(in_buf, in_buf_id);

        if (frame_id % _num_gpu_frames == 0) {
            uint8_t * out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_id);
            header = (struct rawGPUFrameHeader *)out_frame;
            output = (int32_t *)(&out_frame[sizeof(struct rawGPUFrameHeader)]);

            header->fpga_seq_num = get_fpga_seq_num(in_buf, in_buf_id);
            uint16_t stream_id = get_stream_id(in_buf, in_buf_id);
            stream_id_t s_stream_id = extract_stream_id(stream_id);
            s_stream_id.crate_id = 2; // TODO set this to match the number of crate-pairs active.
            header->stream_id = encode_stream_id(s_stream_id);
            timeval time_v = get_first_packet_recv_time(in_buf, in_buf_id);
            header->epoch_time_sec = (uint32_t)time_v.tv_sec;
            header->epoch_time_usec = (uint32_t)time_v.tv_usec;
            header->unused = 2048;

            header->lost_frames = get_lost_timesamples(in_buf, in_buf_id);

            for (int i = 0; i < in_buf->frame_size/sizeof(int32_t); ++i) {
                output[i] = input[i];
            }

        } else {
            header->lost_frames += get_lost_timesamples(in_buf, in_buf_id);

            for (int i = 0; i < in_buf->frame_size/sizeof(int32_t); ++i) {
                output[i] += input[i];
            }
        }

        // TODO This requires a new meta data object with the summed properties of
        // all the objects that went into it.
        //pass_metadata(in_buf, in_buf_id, out_buf, out_buf_id);

        mark_frame_empty(in_buf, unique_name.c_str(), in_buf_id);
        in_buf_id = (in_buf_id + 1) % in_buf->num_frames;
        frame_id++;

        if (frame_id % _num_gpu_frames == 0) {
            mark_frame_full(out_buf, unique_name.c_str(), out_buf_id);
            out_buf_id = (out_buf_id + 1) % out_buf->num_frames;
        }
    }
}
