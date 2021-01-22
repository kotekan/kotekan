#include "mergeRawBuffer.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"  // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"          // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp" // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp" // for INFO

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <visUtil.hpp> // for frameID, modulo
#include <iostream>    // for cout

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(mergeRawBuffer);

mergeRawBuffer::mergeRawBuffer(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&mergeRawBuffer::main_thread, this)) {

    // Apply config
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    
    INFO("Samples_per_data11 {:d} frame_size {:d}", _samples_per_data_set, (uint32_t)in_buf->frame_size);

//    if (_samples_per_data_set != (uint32_t)in_buf->frame_size) {
//        throw std::runtime_error("compressLostSamples in_frame has the wrong size.");
//    } 

}

mergeRawBuffer::~mergeRawBuffer() {}


void mergeRawBuffer::main_thread() {
    // frameID in_buffer_ID(in_buf);
    frameID frame_id(in_buf);
    uint8_t* in_frame;
    // frameID out_buffer_ID(out_buf);
    // uint32_t* out_frame;

    
    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (in_frame == nullptr)
            break;

	BeamMetadata* metadata = (BeamMetadata*)get_metadata(in_buf, frame_id);
	INFO("Test1\n");

        // out_frame = (uint32_t*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
        // if (out_frame == nullptr)
        //     break;
        
        //const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();

        //std::string frequency_bins = "";
        //for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
        //    frequency_bins +=
        //        fmt::format("{:d}", Telescope::instance().to_freq_id(metadata->stream_id, f));
        //    if (f != num_freq_per_stream - 1)
        //        frequency_bins += ", ";
        //}

        //INFO("Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq_bins: {:s}, first value: {:d}+{:d}i",
        //     metadata->ra, metadata->dec, metadata->scaling, frequency_bins, in_frame[0] & 0x0F,
        //     (in_frame[0] & 0xF0) >> 4);

	//uint32_t total_frames = 0;
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

	frame_id++;
    }
}
