#include "sortBeamBuffer.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp"  // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp" // for INFO

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <iostream>    // for cout
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <visUtil.hpp> // for frameID, modulo

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(sortBeamBuffer);

sortBeamBuffer::sortBeamBuffer(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&sortBeamBuffer::main_thread, this)) {
	
    // Apply config
    has_freq_bin = config.get_default<uint32_t>(unique_name, "has_freq_bin", false);
    num_freq = config.get_default<uint32_t>(unique_name, "num_freq", 1024);
    time_frames_per_dump = config.get_default<uint32_t>(unique_name, "time_frames_per_dump", 1);
    freq_chans_per_dump = config.get_default<uint32_t>(unique_name, "freq_chans_per_dump", 1024);
    num_freq_per_sorted_frame = config.get<uint32_t>(unique_name, "num_freq_per_sorted_frame");
    num_time_per_sorted_frame = config.get<uint32_t>(unique_name, "num_time_per_sorted_frame");
    samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    num_dump_time_frame = config.get_default<uint32_t>(unique_name, "num_dump_time_frame", 1);
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

sortBeamBuffer::~sortBeamBuffer() {}


void sortBeamBuffer::main_thread(){
    frameID in_buffer_ID(in_buf);
    uint8_t* in_frame = nullptr;
    frameID out_buffer_ID(out_buf);
    uint8_t* out_frame = nullptr;
    int32_t freq_bin;
    int32_t frame_nr; // The frame namber
    int64_t offset;
    int64_t fpga_seq_start;
    int64_t frame0 = -9999;
    timespec frame_time;

    /// Init the sort queue
    sort_que = (uint8_t ***)malloc(num_time_per_sorted_frame * sizeof(uint8_t**));
    for (uint32_t i = 0; i < num_time_per_sorted_frame; i++) {
        sort_que[i] = (uint8_t **)malloc(num_freq * sizeof(uint8_t*));
	for (uint32_t j = 0; j < num_freq; j++){
	    sort_que[i][j] = nullptr;
	}
    }
    // Since of the BeamMeta data with frequency ID.
    uint32_t FreqIDBeamMeta_size = sizeof(FreqIDBeamMetadata);

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            break;

	// Get the metadata for each frame. 
	if (has_freq_bin) {
	    FreqIDBeamMetadata* in_metadata = (FreqIDBeamMetadata*)get_metadata(in_buf, in_buffer_ID);
            freq_bin = in_metadata -> frequency_bin;
	    fpga_seq_start = in_metadata -> fpga_seq_start;
        }	
	else {
	    BeamMetadata* in_metadata = (BeamMetadata*)get_metadata(in_buf, in_buffer_ID);
	    fpga_seq_start = in_metadata -> fpga_seq_start;
	    const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();
	    for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                freq_bin = Telescope::instance().to_freq_id(in_metadata->stream_id, f);
            }

        }        
	// add frame to the sort queue
	frame_nr = fpga_seq_start / samples_per_data_set;
	// When the recording just started
	if (frame0 == -9999){
	    frame0 = frame_nr;
	}
	// index offset
	offset = frame_nr - frame0;
	// Add to queue
	if (offset >= 0){ // When the coming frame is after the time frame0, which is correct
	    if (offset < num_time_per_sorted_frame){ // frame comes before the frame waiting limit.
	        sort_que[offset][freq_bin] = in_frame;
	    }
	    else if (offset == num_time_per_sorted_frame){ 
		// Frame comes to the end of the queue. Dump the data to the merged buffer.
		// Get a new frame from out buffer.
                out_frame = (uint8_t*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
		if (out_frame == nullptr)
                    break;
                allocate_new_metadata_object(out_buf, out_buffer_ID);
                MergedBeamMetadata* out_metadata =
                    (MergedBeamMetadata*)get_metadata(out_buf, out_buffer_ID);
		// populate the merged_buffer/out_buffer frame header.
		// TODO maybe add the option what to dump out.
                	        
                
                out_metadata->sub_frame_pre_frame = freq_chans_per_dump * time_frames_per_dump;
                out_metadata->sub_frame_metadata_size = FreqIDBeamMeta_size;
                out_metadata->sub_frame_data_size = in_buf->frame_size;
		// Dump the data for all the frequency
	    }
	    else{
		// Frame comes beyong the waiting frame limit, this general should not happen.
		INFO("Frame {:d} is beyond the queue waiting limit", fpga_seq_start);
	    }
	}
	else{ // A frame come after that time has dumped out. 
            INFO("Frame {:d} comes too late", fpga_seq_start);
	}
        
	// When the queue offset is in the end, dump the first frame. 
	
        // Roll the queue, so the second line of time frame move to the first.
	
    }



}
