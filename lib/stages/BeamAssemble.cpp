// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
* @file   BeamAssemble.cpp
* @brief  This file implements the BeamAssemble stage
*
* @author Mehdi Najafi
* @date   28 AUG 2022
*****************************************************/

#include "BeamAssemble.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for Telescope
#include "buffer.h"           // for get_metadata, mark_frame_empty, register_consumer, wait_fo...
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo, ts_to_double
#include "FrequencyAssembledMetadata.hpp"  // for FrequencyBin

#include "fmt.hpp" // for format

#include <map>      // for std::map
#include <stdint.h> // for uint8_t, uint32_t
#include <utility>  // for std::make_pair
#include <functional> // for std::pair

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(BeamAssemble);

STAGE_CONSTRUCTOR(BeamAssemble) {
    // retrieve th input buffer and register this class as a consumer for it
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    // retrieve the output buffer and register this class as a producer for it
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    // retrieve the expected number of frequencies in the output
    num_freq_per_output_frame = config.get<uint32_t>(unique_name, "num_freqs");
    // retrieve the timeout in seconds for the arriving data in input buffer frames
    arriving_data_timeout = config.get<int32_t>(unique_name, "time_out");
}

BeamAssemble::~BeamAssemble() {}

void BeamAssemble::main_thread() {

    frameID in_frame_id(in_buf), out_frame_id(out_buf);
    uint8_t *in_frame, *out_frame;

    // the frame map used for timeout controlled frames
    std::map<int64_t, std::pair<uint8_t*, int>, std::greater<int>> out_frame_map;

    // timestamp used for the frame map control
    int64_t current_frame_timestamp;

    // allocate new metadata objects for all the output buffer frames
    for (int i = 0; i < out_buf->num_frames; i++) {
        allocate_new_metadata_object(out_buf, i);
    }

    // thread infinite loop
    while (!stop_thread) {

        // since this constant may change during run-time, keep it here
        const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();

        // acquire a frame from the input buffer
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;

        // retrieve the metadata and create a new metadata based on it
        BeamMetadata* metadata = (BeamMetadata*)get_metadata(in_buf, in_frame_id);

        // get the current timestamp in milliseconds from the current input frame metadata
        current_frame_timestamp = timespec_to_milliseconds(metadata->gps_time);

        // loop over the open output frames and release the output frames if the timeout has reached
        for (auto it = out_frame_map.begin(); it != out_frame_map.end();) {
            // check if the time has passed more than arriving_data_timeout since this output frame
            // has been opened
            if (current_frame_timestamp - it->first > arriving_data_timeout) {
                // release the output frame in the map
                mark_frame_full(out_buf, unique_name.c_str(), it->second.second);
                // remove the output frame element from the map
                out_frame_map.erase(it++);
            }
            else {
                it++;
            }
        }

        // check if the timestamp corresponds to the past time frame which is no longer considered/accepted
        if ( out_frame_map.size() && current_frame_timestamp + arriving_data_timeout < out_frame_map.begin()->first)
        {
            // log the receiving outdated frame frequency
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                // get the frequency id
                auto freq_id = Telescope::instance().to_freq_id(metadata->stream_id, f);
                // print info about the late beam
                printf( fmt::format("LATE Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq: {:d}, first value: {:d}+{:d}i\n",
                        metadata->ra, metadata->dec, metadata->scaling, freq_id, 
                        in_frame[0] & 0x0F, (in_frame[0] & 0xF0) >> 4).c_str() );
            }
        }
        else {
            // determine which output frame to write on based on the timestamp: if this is a new time
            // stamp, make a new entry in the map and acquire an empty frame from the output buffer
            if (out_frame_map.find(current_frame_timestamp) == out_frame_map.end()) {
                // wait for the first available empty frame
                out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
                if (out_frame == nullptr) {
                    // no output frame buffer available, so leave the thread infinite loop
                    break;
                }

                // allocate a new metadata object for this empty output buffer frame
                allocate_new_metadata_object(out_buf, out_frame_id);

                // copy the base metadata to this frame metadata
                copy_base_to_frequency_assembled_Metadata( metadata, get_metadata(out_buf, out_frame_id));

                // add the new output frame to the map
                out_frame_map[current_frame_timestamp] = std::make_pair( out_frame, out_frame_id);
                out_frame_id++;

                // zero all the frame: zeros implies missing data as well
                memset(out_frame, 0, out_buf->frame_size);
            } else {
                // retrieve the output frame from the map
                out_frame = out_frame_map[current_frame_timestamp].first;
            }

            // loop over received frequencies in the input buffer frame and
            // put their data in the corresponding output buffer frame
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                // get the frequency id
                auto freq_id = Telescope::instance().to_freq_id(metadata->stream_id, f);

                // input frame buffer frequency data size
                auto in_frame_freq_size = in_buf->frame_size / num_freq_per_stream;

                // determine the offset based on the frequency id
                auto out_frame_offset = freq_id * (out_buf->frame_size - num_freq_per_output_frame) / num_freq_per_output_frame;
                // is that the same as: freq_id * in_frame_freq_size ?
                // assert(out_frame_offset == freq_id * in_frame_freq_size);

                // copy the data from in_frame to the out_frame + some offset with no change
                // Note that the first num_freq_per_output_frame is the frequency id indicator,
                // which is 0 or 1 indicating the frequency data copied to the output
                memcpy(out_frame + num_freq_per_output_frame + out_frame_offset, in_frame, in_frame_freq_size);
                // turn on the frequency id indicator, located at the beginning of the output frame
                out_frame[freq_id] = 1;

                // also inform the frequency bin added to the metadata about freq_id
                FrequencyBin* freq_bin = (FrequencyBin*)get_metadata(out_buf, out_frame_map[current_frame_timestamp].second);
                frequencyBin_add_frequency_id(freq_bin, freq_id);

    #ifdef DEBUGGING
                // print out some metadata and the first element of frame data
                printf( fmt::format("Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq_bins: {:d} in [{:d},{:d}], first value: {:d}+{:d}i\n",
                        metadata->ra, metadata->dec, metadata->scaling, freq_id, 
                        freq_bin->lower_band_frequency, freq_bin->higher_band_frequency, 
                        in_frame[0] & 0x0F, (in_frame[0] & 0xF0) >> 4).c_str() );
    #endif
            }
        }

        // TODO later: add some statistics

        // release the input frame and mark it as empty
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id++;
    }

    // Be a nice thread: loop over all the open output buffer frames and release them
    for (auto it = out_frame_map.begin(); it != out_frame_map.end();) {
        // release the output frame in the map
        mark_frame_full(out_buf, unique_name.c_str(), it->second.second);
        // remove the output frame element from the map
        out_frame_map.erase(it++);
    }
}
