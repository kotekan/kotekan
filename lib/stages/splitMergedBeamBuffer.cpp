#include "splitMergedBeamBuffer.hpp"

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

REGISTER_KOTEKAN_STAGE(splitMergedBeamBuffer);

splitMergedBeamBuffer::splitMergedBeamBuffer(Config& config_, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&splitMergedBeamBuffer::main_thread, this)) {

    // Apply config
    //_samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    //_num_pol = config.get<uint32_t>(unique_name, "num_pol");
    //_raw_frames_per_merged_frame = config.get<uint32_t>(unique_name,
    //"raw_frames_per_merged_frame");
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");

    register_producer(out_buf, unique_name.c_str());
}

splitMergedBeamBuffer::~splitMergedBeamBuffer() {}

void splitMergedBeamBuffer::main_thread() {
    frameID in_buffer_ID(in_buf);
    uint8_t* in_frame;
    frameID out_buffer_ID(out_buf);
    uint8_t* out_frame;
    uint32_t sub_frame_matedata_size;
    uint32_t sub_frame_data_size;
    uint32_t sub_frame_start;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            break;

        MergedBeamMetadata* in_metadata = (MergedBeamMetadata*)get_metadata(in_buf, in_buffer_ID);
        DEBUG2("Number of subframes {:d}, subframe metasize {:d}, subframe data size {:d}:\n",
               in_metadata->sub_frame_pre_frame, in_metadata->sub_frame_metadata_size,
               in_metadata->sub_frame_data_size);

        sub_frame_matedata_size = in_metadata->sub_frame_metadata_size;
        sub_frame_data_size = in_metadata->sub_frame_data_size;
        for (uint32_t i = 0; i < in_metadata->sub_frame_pre_frame; i++) {
            // Compute the merged frame position to unpack the merged frames
            sub_frame_start = (sub_frame_matedata_size + sub_frame_data_size) * i;
            // Get an empty out frame
            out_frame = (uint8_t*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
            if (out_frame == nullptr)
                break;

            // Allocated the new metadata for the outpug buffer
            allocate_new_metadata_object(out_buf, out_buffer_ID);
            BeamMetadata* out_metadata = (BeamMetadata*)get_metadata(out_buf, out_buffer_ID);
            // Get pointer to the beam metadata with frequency
            FreqIDBeamMetadata* sub_frame_metadata =
                (FreqIDBeamMetadata*)&in_frame[sub_frame_start];
            // Fill up the out put buffer metadata.
            out_metadata->fpga_seq_start = sub_frame_metadata->fpga_seq_start;
            out_metadata->ctime = sub_frame_metadata->ctime;
            out_metadata->stream_id = sub_frame_metadata->stream_id;
            out_metadata->dataset_id = sub_frame_metadata->dataset_id;
            out_metadata->beam_number = sub_frame_metadata->beam_number;
            out_metadata->ra = sub_frame_metadata->ra;
            out_metadata->dec = sub_frame_metadata->dec;
            out_metadata->scaling = sub_frame_metadata->scaling;
            uint8_t* data_start = &in_frame[sub_frame_start + sub_frame_matedata_size];
            // copy data from in frame to out_frame/single frame
            memcpy(out_frame, data_start, sub_frame_data_size);
            mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);
            out_buffer_ID++;
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        in_buffer_ID++;
    }
}
