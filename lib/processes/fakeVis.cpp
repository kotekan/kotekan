#include "fakeVis.hpp"

fakeVis::fakeVis(Config &config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visTransform::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int("/", "num_elements");
    block_size = config.get_int("/", "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    // TODO: num_prod?

    // Get the output buffer
    std::string buffer_name = config.get_string_array(unique_name, "output_buffer");

    // Fetch the buffer, register it
    output_buffer = buffer_container.get_buffer(buffer_name);
    register_consumer(output_buffer, unique_name.c_str());

    // Get frequency IDs from config
    // TODO: check casting works here
    for (int32_t f : config.get_int_array(unique_name, "freq")) {
        freq.push_back((uint16_t) f)
    }

}

void fakeVis::apply_config(uint64_t fpga_seq) {

}

void fakeVis::main_thread() {

    uint8_t * frame = nullptr;
    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    while (!stop_thread) {

       // Wait for the buffer to be filled with data
       if(wait_for_empty_frame(output_buffer, unique_name.c_str(),
                               output_frame_id) == nullptr) {
           break;
       }

       // Allocate metadata and get frame
       allocate_new_metadata_object(buf, output_frame_id);
       auto output_frame = visFrameView(output_buffer, output_frame_id,
                                        num_elements, num_eigenvectors);

       // Below taken from visWriter

       // TODO: dataset ID properly when we have gated data
       output_frame.dataset_id() = 0;

       // Set the frequency index from the stream id of this buffer
       stream_id_t stream_id = get_stream_id_t(buf, frame_id);
       output_frame.freq_id() = bin_number_chime(&stream_id);

       // Set the time
       // TODO: get the GPS time instead
       uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
       timeval tv = get_first_packet_recv_time(buf, frame_id);
       timespec ts;
       TIMEVAL_TO_TIMESPEC(&tv, &ts);
       output_frame.time() = std::make_tuple(fpga_seq, ts);

       // TODO: do something with the list timesamples data
       // uint64_t lost_samples = get_lost_timesamples(buf, frame_id);

       // Copy the visibility data into a proper triangle and write into
       // the file
       copy_vis_triangle((int32_t *)frame, input_remap, block_size,
                         num_elements, output_frame.vis());

       // Mark the buffers and move on
       mark_frame_empty(buf, unique_name.c_str(), frame_id);
       mark_frame_full(output_buffer, unique_name.c_str(),
                       output_frame_id);

       // Advance the current frame ids
       std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
       output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
       buf_ind++;


        }
    }
}
