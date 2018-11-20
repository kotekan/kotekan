#include "visMerge.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "errors.h"
#include "processFactory.hpp"
#include "visUtil.hpp"

#include <stdint.h>
#include <atomic>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <tuple>

REGISTER_KOTEKAN_PROCESS(visMerge);


visMerge::visMerge(Config& config,
                   const std::string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visMerge::main_thread, this)) {

    // Setup the output vector
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the list of buffers that this process shoud connect to
    std::vector<std::string> input_buffer_names =
        config.get<std::vector<std::string>>(unique_name, "in_bufs");

    // Fetch the input buffers, register them, and store them in our buffer vector
    for(auto name : input_buffer_names) {
        auto buf = buffer_container.get_buffer(name);

        if(buf->frame_size > out_buf->frame_size) {
            throw std::invalid_argument("Input buffer [" + name +
                                        "] larger that output buffer size.");
        }

        register_consumer(buf, unique_name.c_str());
        in_bufs.push_back({buf, 0});
    }

}

void visMerge::main_thread() {

    struct Buffer* buf;
    unsigned int frame_id = 0;
    unsigned int output_frame_id = 0;

    while (!stop_thread) {

        // This is where all the main set of work happens. Iterate over the
        // available buffers, wait for data to appear and then attempt to write
        // the data into a file
        for(auto& buffer_pair : in_bufs) {
            std::tie(buf, frame_id) = buffer_pair;

            // Calculate the timeout
            auto timeout = double_to_ts(current_time() + 0.1);

            // Find the next available buffer
            int status = wait_for_full_frame_timeout(buf, unique_name.c_str(),
                                                     frame_id, timeout);
            if(status == 1) continue;  // Timed out, try next buffer
            if(status == -1) break;  // Got shutdown signal

            // Wait for the buffer to be filled with data
            if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                    output_frame_id) == nullptr) {
                break;
            }

            DEBUG("Merging buffer %s[%i] into %s[%i]",
                  buf->buffer_name, frame_id,
                  out_buf->buffer_name, output_frame_id);

            // Transfer metadata
            pass_metadata(buf, frame_id, out_buf, output_frame_id);

            // Copy the frame data here:
            std::memcpy(out_buf->frames[output_frame_id],
                        buf->frames[frame_id],
                        buf->frame_size);

            // Mark the buffers and move on
            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            mark_frame_full(out_buf, unique_name.c_str(),
                            output_frame_id);

            // Advance the current frame ids
            std::get<1>(buffer_pair) = (frame_id + 1) % buf->num_frames;
            output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        }

    }

}