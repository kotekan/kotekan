#include "bufferDelay.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO, DEBUG2, FATAL_ERROR
#include "visUtil.hpp"

#include "fmt.hpp" // for format, fmt
#include "json.hpp"

#include <algorithm>  // for max
#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error, invalid_argument
#include <stdint.h>   // for uint8_t
#include <string.h>   // for memcpy


using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferDelay);

bufferDelay::bufferDelay(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferDelay::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    
    _copy_frame = config.get_default<bool>(unique_name, "copy_frame", false);
    _num_frames_to_hold = config.get<uint32_t>(unique_name, "num_frames_to_hold");

    if (_num_frames_to_hold > (uint32_t)in_buf->num_frames) {
      throw std::invalid_argument(fmt::format(fmt("The no. of frames to hold ({:d}) is greater than the number of frames that can be stored in the input buffer ({:d})."),
            _num_frames_to_hold, in_buf->num_frames));
    }

}

void bufferDelay::main_thread() {

    frameID in_frame_id(in_buf);
    frameID in_frame_release_id(in_buf);
    frameID out_frame_id(out_buf);
    uint32_t in_frame_hold_ctr = 0;

    uint8_t* output_frame = nullptr;

    while (!stop_thread) {

        uint8_t* input_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (input_frame == nullptr)
            return;
        
        DEBUG("Got new input frame {:d}", in_frame_id);

        // If the in buffer is holding the right amount of frames, release one
        if (in_frame_hold_ctr >= _num_frames_to_hold) {

          // Get a new output frame
          output_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
          if (output_frame == nullptr)
            return;
 
          // Buffer sizes must match exactly
          if (in_buf->frame_size != out_buf->frame_size) {
            throw std::runtime_error(
                fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                  in_buf->frame_size, out_buf->frame_size));
          }

          if (_copy_frame) {
            allocate_new_metadata_object(out_buf, out_frame_id);
            // Metadata sizes must match exactly
            if (in_buf->metadata[in_frame_release_id]->metadata_size
                != out_buf->metadata[out_frame_id]->metadata_size) {
              throw std::runtime_error(
                  fmt::format(fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                    in_buf->metadata[in_frame_release_id]->metadata_size,
                    out_buf->metadata[out_frame_id]->metadata_size));
            }
            copy_metadata(in_buf, in_frame_release_id, out_buf, out_frame_id);
            std::memcpy(output_frame, in_buf->frames[in_frame_release_id], in_buf->frame_size);
          } else {
            pass_metadata(in_buf, in_frame_release_id, out_buf, out_frame_id);
            swap_frames(in_buf, in_frame_release_id, out_buf, out_frame_id);
          }

          DEBUG("Reached maximum no. of frames to hold. Releasing oldest frame... in_frame_id: {:d}, in_frame_hold_ctr: {:d}, in_frame_release_id: {:d}", in_frame_id, in_frame_hold_ctr, in_frame_release_id);

          mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
          out_frame_id++;
         
          // Release the input buffer frame.
          mark_frame_empty(in_buf, unique_name.c_str(), in_frame_release_id);
          in_frame_release_id++;
        }
        else
          in_frame_hold_ctr++;

        // Increase the in_frame_id for the input buffer
        in_frame_id++;
          
        DEBUG("Holding {:d} frames", in_frame_hold_ctr);

    }

}
