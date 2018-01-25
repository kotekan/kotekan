#include "freqSlicer.hpp"
#include "visFile.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"
#include "util.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "gpuPostProcess.hpp"
#include "errors.h"
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>


// This is the implementation of the constructor for freqSplit
// This is colon means the initialization list follows. It is executed 
// before the body of the constructor.
freqSplit::freqSplit(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) : 
    
    // Initialization list. It calls the constructor for the base class
    // TODO: I still don't understand the last argument.
    // It binds the freqSplit main_thread to 'this' KotekanProcess object??
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&freqSplit::main_thread, this)) 

// Here is where the body of the freqSplit constructor starts.
// It was very confusing because the bracket was in front of the base class
// constructor and it looked like the implementation for that!!
{
    // Fetch any simple configuration
    // This is how to get values from the config file:
    // config.get_type("level in config file","name of variable")

// TODO: delete. I don't think I need those
//    num_elements = config.get_int("/", "num_elements");
//    block_size = config.get_int("/", "block_size");
//    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> output_buffer_names =
        config.get_string_array(unique_name, "output_buffers");

    // Fetch the output buffers, register them, and store them in our buffer vector
    for(auto name : output_buffer_names) {
        auto buf = buffer_container.get_buffer(name);
        register_producer(buf, unique_name.c_str());
        output_buffers.push_back({buf, 0});// push_back appends to the list
    }

    // Setup the input buffer
    input_buffer = get_buffer("input_buffer");
    register_consumer(input_buffer, unique_name.c_str());

    // I don't think I need this
    // Array to reorder inputs from buffer order to visibility order?
    // Get the indices for reordering
//    input_remap = std::get<0>(parse_reorder_default(config, unique_name));
}

// These are the two member functions implemented:
//
void freqSplit::apply_config(uint64_t fpga_seq) {

}

void freqSplit::main_thread() {

    uint8_t * frame = nullptr;
    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    unsigned int buf_ind;

    while (!stop_thread) {

        // This is where all the main set of work happens. Iterate over the
        // available buffers, wait for data to appear and then attempt to write
        // the data into a file
        

        // Wait for the input buffer to be filled with data
        // I think you can assign the frame on the fly because it
        // exists already in memory (unlike the output frame)
        if((input_frame = wait_for_full_frame(input_buffer, 
                    unique_name.c_str(),input_frame_id)) == nullptr) {
            break;
        }

        // frequency index of this frame
        freq = input_frame.freq_id();

        // After figuring out which frequency I have, I get the right
        // output buffer in this way:
        if(freq < 512) {
            buf_ind=0;
        } else {
            buf_ind=1;
        }
        auto& buffer_pair = output_buffers[buf_ind];
        std::tie(buf, frame_id) = buffer_pair; 

        INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(buf, unique_name.c_str(),
                                        frame_id) == nullptr) {
            break;
        }


        // TODO: I think this allocates memory space for the new
        // metadata. I don't think I need this here because I
        // am going to create a pointer copy to the previous 
        // frame's metadata pointer. Need to check.
        allocate_new_metadata_object(buf, frame_id);

        // Copy frame and create view
        auto frame = visFrameView(buf, frame_id, input_frame);


        // TODO: delete. this was substituted by the previous one
        // Pick specific frame in output buffer?
//        auto frame = visFrameView(buf, frame_id, num_elements,
//                                                num_eigenvectors);

        // TODO: Dlete. I think I don't need this either, because
        // this is part of the metadata, which will be 
        // pointer-copied in it's entirety.
//        frame.dataset_id() = 0; //Marks it as chime data...

        // Mark the buffers and move on
        mark_frame_empty(input_buffer, unique_name.c_str(), 
                                                input_frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
    }
        








//void freqSplit::main_thread() {
//
//    uint8_t * frame = nullptr;
//    struct Buffer* buf;
//    unsigned int frame_id = 0;
//    unsigned int input_frame_id = 0;
//
//    while (!stop_thread) {
//
//        // This is where all the main set of work happens. Iterate over the
//        // available buffers, wait for data to appear and then attempt to write
//        // the data into a file
//        unsigned int buf_ind = 0;
//        // * This is a Range-based for loop in c++. It iterates over 
//        // all possible values of the vector output_buffers
//        // the values are available in the iteration variable buffer_pair
//        // * auto means that the type of buffer_pair will be infered
//        // from the assignment (in this case the type of each element
//        // of output_buffers.
//        // * the & means buffer_pair is a reference so changes to it
//        // will affect the original vector.
//        for(auto& buffer_pair : output_buffers) { 
//            std::tie(buf, frame_id) = buffer_pair;
//
//            INFO("Buffer %i has frame_id=%i", buf_ind, frame_id);
//
//            // Wait for the buffer to be filled with data
//            if((frame = wait_for_empty_frame(buf, unique_name.c_str(),
//                                            frame_id)) == nullptr) {
//                break;
//            }
//
//            // Wait for the buffer to be filled with data
//            if(wait_for_full_frame(input_buffer, unique_name.c_str(),
//                                    input_frame_id) == nullptr) {
//                break;
//            }
//
//            // If this really initializes memory in output buffer
//            // Than I should keep it as output buffer (buf) and 
//            // (frame_id)
//            allocate_new_metadata_object(input_buffer, input_frame_id); //initializes memory in output buffer
//
//            auto input_frame = visFrameView(input_buffer, input_frame_id,
//                                             num_elements, num_eigenvectors); //pick specific frame in output buffer
//
//            // TODO: set the dataset ID properly when we have gated data
//            input_frame.dataset_id() = 0; //Marks it as chime data...
//
//
//
//            // Set the frequency index from the stream id of this buffer
//            stream_id_t stream_id = get_stream_id_t(buf, frame_id);
//            input_frame.freq_id() = bin_number_chime(&stream_id);
//            //I get 4 freqs (each is a stream) and need to assign freq numbers
//
//
//
//
//            // Set the time
//            // TODO: get the GPS time instead
//            uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
//            timeval tv = get_first_packet_recv_time(buf, frame_id);
//            timespec ts;
//            TIMEVAL_TO_TIMESPEC(&tv, &ts);
//            input_frame.time() = std::make_tuple(fpga_seq, ts);
//
//            // TODO: do something with the list timesamples data
//            // uint64_t lost_samples = get_lost_timesamples(buf, frame_id);
//
//            // Copy the visibility data into a proper triangle and write into
//            // the file
//            copy_vis_triangle((int32_t *)frame, input_remap, block_size,
//                              num_elements, input_frame.vis());
//
//            // Mark the buffers and move on
//            mark_frame_empty(buf, unique_name.c_str(), frame_id);
//            mark_frame_full(input_buffer, unique_name.c_str(),
//                            input_frame_id);
//
//            // Advance the current frame ids
//            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
//            input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
//            buf_ind++;
//        }
//
//    }
//
//}


