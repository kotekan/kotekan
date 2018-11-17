#include "mergeBuffer.hpp"

REGISTER_KOTEKAN_PROCESS(mergeBuffer);

mergeBuffer::mergeBuffer(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&mergeBuffer::main_thread, this)) {

    in_bufs = get_buffer_array("in_bufs");
    for (struct Buffer * in_buf : in_bufs)
        register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

mergeBuffer::~mergeBuffer() {
}

void mergeBuffer::apply_config(uint64_t fpga_seq) {
}

void mergeBuffer::main_thread() {

    std::vector<int> input_frame_ids(in_bufs.size());
    std::fill(input_frame_ids.begin(), input_frame_ids.end(), 0);

    int output_frame_id = 0;

    while(!stop_thread) {
        for (int32_t i = 0; i < in_bufs.size(); ++i) {
            int * input = (int *)wait_for_full_frame(in_bufs.at(i), unique_name.c_str(), input_frame_ids.at(i));
            if (input == NULL) break;
            int * output = (int *)wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id);
            if (output == NULL) break;

            swap_frames(in_bufs.at(i), input_frame_ids.at(i), out_buf, output_frame_id);

            pass_metadata(in_bufs.at(i), input_frame_ids.at(i), out_buf, output_frame_id);

            mark_frame_empty(in_bufs.at(i), unique_name.c_str(), input_frame_ids.at(i));
            mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);

            input_frame_ids.at(i) = (input_frame_ids.at(i) + 1) % in_bufs.at(i)->num_frames;
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }
    }
}
