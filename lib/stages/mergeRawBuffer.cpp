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
    _num_pol = config.get<uint32_t>(unique_name, "num_pol");
    _raw_frames_per_merged_frame = config.get<uint32_t>(unique_name, "raw_frames_per_merged_frame"); 
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");

    register_producer(out_buf, unique_name.c_str());
    
}

mergeRawBuffer::~mergeRawBuffer() {}


void mergeRawBuffer::main_thread() {
    frameID in_buffer_ID(in_buf);
    uint8_t* in_frame;
    frameID out_buffer_ID(out_buf);
    uint8_t* out_frame;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            break;

	BeamMetadata* in_metadata = (BeamMetadata*)get_metadata(in_buf, in_buffer_ID);

        out_frame = (uint8_t*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
        if (out_frame == nullptr)
             break;
        
        // compute the size of the sub_frame of the out_frame
	uint32_t sub_frame_index = in_buffer_ID % _raw_frames_per_merged_frame; 
	uint32_t sub_frame_size = sizeof(BeamMetadata) + in_buf -> frame_size;
	// Get the start position of a sub_out_frame's meatadata
	uint32_t out_frame_metadata_pos = sub_frame_size * sub_frame_index;
	// Get the memory block for the metadata of sub_out_frame.
	BeamMetadata* sub_frame_metadata = (BeamMetadata *)&out_frame[out_frame_metadata_pos];
	// Copy the in_frame metadata the the out frame's sub_out_frame.
        memcpy(in_metadata, sub_frame_metadata, sizeof(BeamMetadata));
	// Get the start position of a sub_out_frame's data
	uint32_t out_frame_data_pos = out_frame_metadata_pos + sizeof(BeamMetadata);
	// Get the memory block for the data of of a subframe.
	uint8_t * sub_frame_data = &out_frame[out_frame_data_pos];
	// copy data from in frame to sub_out_frame
	memcpy(in_frame, sub_frame_data, in_buf -> frame_size);
	std::cout<<"BeamMetadata size:" <<sizeof(BeamMetadata) << std::endl;
        

	BeamMetadata* debud_frame_metadata = (BeamMetadata *)&out_frame[sub_frame_size];
	DEBUG2("Sub frame Beam RA: {:f}, Dec: {:f}, scaling: {:d}, , beam_num: {:d}", debud_frame_metadata -> ra, 
	       debud_frame_metadata -> dec, debud_frame_metadata -> scaling, debud_frame_metadata-> beam_number);
        //INFO("InBuff: Beam RA: {:f}, Dec: {:f}, scaling: {:d}, beam_num: {:d}, \n outBuff: Beam RA: {:f}, Dec: {:f}, scaling: {:d}, beam_num: {:d}",
        //     in_metadata->ra, in_metadata->dec, in_metadata->scaling, in_metadata->beam_number, 
	//     out_metadata->ra, out_metadata->dec, out_metadata->scaling, out_metadata->beam_number);
        
	// INFO("In Buffer: Tenth value: {:d}+{:d}i \n Out Buffer: Tenth value: {:d}+{:d}i", 	in_frame[9] & 0x0F, (in_frame[9] & 0xF0) >> 4, out_frame[9] & 0x0F, (out_frame[9] & 0xF0) >> 4);
        
	const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();
	std::string frequency_bins = "";
        for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
            frequency_bins +=
                fmt::format("{:d}", Telescope::instance().to_freq_id(in_metadata->stream_id, f));
            if (f != num_freq_per_stream - 1)
                frequency_bins += ", ";
        }
	//INFO("In Buffer: stream_id: {:s}, freq_bins: {:s}",in_metadata->stream_id, frequency_bins);
        //INFO("frame type {:S}", typeid(in_frame).name().c_str());
	//uint32_t total_frames = 0;
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID++);
	if (sub_frame_index == _raw_frames_per_merged_frame - 1){
            mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID++);
	}
	INFO("In buffer ID {:d}, out buffer ID {:d}", in_buffer_ID, out_buffer_ID);
    }
}
