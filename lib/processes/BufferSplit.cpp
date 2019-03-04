#include "BufferSplit.hpp"

REGISTER_KOTEKAN_PROCESS(BufferSplit);

BufferSplit::BufferSplit(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&BufferSplit::main_thread, this)) {

    out_bufs = get_buffer_array("out_bufs");
    for (struct Buffer * out_buf : out_bufs)
        register_producer(out_buf, unique_name.c_str());

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

BufferSplit::~BufferSplit() {
}

void BufferSplit::apply_config(uint64_t fpga_seq) {
}

void BufferSplit::main_thread() {

    std::vector<int> output_frame_ids(out_bufs.size());
    std::fill(output_frame_ids.begin(), output_frame_ids.end(), 0);

    int input_frame_id = 0;

    while(!stop_thread) {
        for (int32_t i = 0; i < out_bufs.size(); ++i) {
            int * input = (int *)wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id);
            if (input == NULL) break;
            int * output = (int *)wait_for_empty_frame(out_bufs.at(i), unique_name.c_str(), output_frame_ids.at(i));
            if (output == NULL) break;

            swap_frames(in_buf, input_frame_id, out_bufs.at(i), output_frame_ids.at(i));

            pass_metadata(in_buf, input_frame_id, out_bufs.at(i), output_frame_ids.at(i));

            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
            mark_frame_full(out_bufs.at(i), unique_name.c_str(), output_frame_ids.at(i));

            input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
            output_frame_ids.at(i) = (output_frame_ids.at(i) + 1) % out_bufs.at(i)->num_frames;
        }
    }
}
