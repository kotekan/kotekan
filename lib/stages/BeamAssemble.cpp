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

#include "BeamMetadata.hpp"               // for BeamMetadata
#include "FrequencyAssembledMetadata.hpp" // for FrequencyBin
#include "StageFactory.hpp"               // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"                  // for Telescope
#include "buffer.h"           // for get_metadata, mark_frame_empty, register_consumer, ...
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo, ts_to_double

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t, uint32_t, uint8_t, int64_t
#include <exception>  // for exception
#include <functional> // for std::make_tuple, std::get
#include <map>        // for std::map
#include <regex>      // for match_results<>::_Base_type
#include <string.h>   // for memcpy, memset
#include <tuple>      // for std::tuple
#include <utility>    // for pair
#include <vector>     // for vector

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
    // check if printout of input beam data is requested
    beam_printout = config.get<bool>(unique_name, "beam_printout");
    // check if printout of late input beam data is requested
    late_beam_printout = config.get<bool>(unique_name, "late_beam_printout");

    // reset the counters
    received_beam_frames_count = accepted_beam_frames_count = missed_beam_frames_count = 0;
}

BeamAssemble::~BeamAssemble() {}

void BeamAssemble::main_thread() {
    // round buffer parameters
    frameID in_frame_id(in_buf), out_frame_id(out_buf);
    uint8_t *in_frame, *out_frame;

    // here I used a frame map for timeout controlled frames, for which
    // I made custom keys using timestamp, Ra, Dec, Scaling. The types
    // are integer and float mixture which requires some custom hash key.
    typedef std::tuple<int64_t, int64_t, int64_t, int64_t> key_t;
    // special structure for the tuple as the map key
    struct key_helper {
        bool operator()(const key_t& v0, const key_t& v1) const {
            return (std::get<0>(v0) > std::get<0>(v1));
        }
        // make a tuple mixture of integers and floats
        static auto make_key(const int64_t i1, const float f1, const float f2, const uint32_t i2) {
            // bitwise float to integral type conversion
            union {
                float from;
                int64_t to;
            } fi1 = {.from = f1}, fi2 = {.from = f2};
            return std::make_tuple(i1, fi1.to, fi2.to, i2);
        }
    };

    // the frame map used for timeout controlled frames
    std::map<key_t, std::tuple<uint8_t*, assembledBeamMetadata*, int>, key_helper> out_frame_map;

    // output buffer metadata
    assembledBeamMetadata* ometadata;

    // timestamp used for the frame map control
    int64_t current_frame_timestamp;

    // thread infinite loop
    while (!stop_thread) {

        // since this constant may change during run-time, keep it here
        const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();

        // acquire a frame from the input buffer
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;

        // increment the number of incoming beam data frames
        received_beam_frames_count++;

        // retrieve the metadata and create a new metadata based on it
        BeamMetadata* metadata = (BeamMetadata*)get_metadata(in_buf, in_frame_id);

        // get the current timestamp in milliseconds from the current input frame metadata
        current_frame_timestamp = timespec_to_milliseconds(metadata->gps_time);

        // loop over the open output frames and release the output frames if the timeout has reached
        for (auto it = out_frame_map.begin(); it != out_frame_map.end();) {
            // check if the time has passed more than arriving_data_timeout since this output frame
            // has been opened
            if (current_frame_timestamp - std::get<0>(it->first) > arriving_data_timeout) {
                // release the output frame in the map
                mark_frame_full(out_buf, unique_name.c_str(), std::get<2>(it->second));
                // retrieve the metadata of the released output buffer frame
                ometadata = std::get<1>(it->second);
                // print out some statistics about the released output buffer frame
                INFO("Released output frame RA: {:f}, Dec: {:f}, scaling: {:d}, freq bin: "
                     "[{:d},{:d}], num_freqs: {:d}\n",
                     ometadata->ra, ometadata->dec, ometadata->scaling,
                     ometadata->lower_band_received_frequency,
                     ometadata->upper_band_received_frequency, ometadata->num_received_frequencies);

                // remove the output frame element from the map
                out_frame_map.erase(it++);
            } else {
                it++;
            }
        }

        // check if the timestamp corresponds to the past time frame which is no longer
        // considered/accepted
        if (out_frame_map.size()
            && current_frame_timestamp + arriving_data_timeout
                   < std::get<0>(out_frame_map.begin()->first)) {
            // increment the number of late/missed incoming beam data frames
            missed_beam_frames_count++;

            // log the receiving outdated frame frequency
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                // get the frequency id
                auto freq_id = Telescope::instance().to_freq_id(metadata->stream_id, f);
                if (late_beam_printout) {
                    // print info about the late beam
                    INFO("LATE Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq: {:d}, first value: "
                         "{:d}+{:d}i\n",
                         metadata->ra, metadata->dec, metadata->scaling, freq_id,
                         in_frame[0] & 0x0F, (in_frame[0] & 0xF0) >> 4);
                }
            }
        } else {
            // increment the number of accepted frame from all incoming beam data frames
            accepted_beam_frames_count++;

            // create a new key for the map, using the current timestamp, Ra, Dec, and Scaling
            auto current_frame_key = key_helper::make_key(current_frame_timestamp, metadata->ra,
                                                          metadata->dec, metadata->scaling);

            // determine which output frame to write on based on the timestamp and RDS:
            // if this is a new time stamp, make a new entry in the map and acquire an
            // empty frame from the output buffer
            if (out_frame_map.find(current_frame_key) == out_frame_map.end()) {
                // wait for the first available empty frame
                out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
                if (out_frame == nullptr) {
                    // no output frame buffer available, so leave the thread infinite loop
                    break;
                }

                // allocate a new metadata object for this empty output buffer frame
                allocate_new_metadata_object(out_buf, out_frame_id);

                // get the new metadata object for the empty output buffer frame
                ometadata = (assembledBeamMetadata*)get_metadata(out_buf, out_frame_id);

                // reset the frequency bin part of the output buffer frame metadata
                frequencyBin_initialize(ometadata);

                // copy the base metadata to this frame metadata
                copy_base_to_frequency_assembled_Metadata(metadata, ometadata);

                // add the new output frame to the map
                out_frame_map[current_frame_key] =
                    std::make_tuple(out_frame, ometadata, out_frame_id);
                out_frame_id++;

                // zero all the frame: zeros implies missing data as well
                memset(out_frame, 0, out_buf->frame_size);
            } else {
                // retrieve the output frame from the map
                out_frame = std::get<0>(out_frame_map[current_frame_key]);
                ometadata = std::get<1>(out_frame_map[current_frame_key]);
            }

            // loop over received frequencies in the input buffer frame and
            // put their data in the corresponding output buffer frame
            for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
                // get the frequency id
                auto freq_id = Telescope::instance().to_freq_id(metadata->stream_id, f);

                // input frame buffer frequency data size
                auto in_frame_freq_size = in_buf->frame_size / num_freq_per_stream;

                // determine the offset based on the frequency id
                auto out_frame_offset = freq_id * (out_buf->frame_size - num_freq_per_output_frame)
                                        / num_freq_per_output_frame;
                // is that the same as: freq_id * in_frame_freq_size ?
                // assert(out_frame_offset == freq_id * in_frame_freq_size);

                // copy the data from in_frame to the out_frame + some offset with no change
                // Note that the first num_freq_per_output_frame is the frequency id indicator,
                // which is 0 or 1 indicating the frequency data copied to the output
                memcpy(out_frame + num_freq_per_output_frame + out_frame_offset,
                       in_frame + f * in_frame_freq_size, in_frame_freq_size);
                // turn on the frequency id indicator, located at the beginning of the output frame
                out_frame[freq_id] = 1;

                // also inform the frequency bin added to the metadata about this freq_id
                frequencyBin_add_frequency_id(ometadata, freq_id);

                if (beam_printout) {
                    // print out some metadata and the first element of frame data
                    INFO("Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq_bins: {:d} in [{:d},{:d}], "
                         "first value: {:d}+{:d}i\n",
                         metadata->ra, metadata->dec, metadata->scaling, freq_id,
                         ometadata->lower_band_received_frequency,
                         ometadata->upper_band_received_frequency, in_frame[0] & 0x0F,
                         (in_frame[0] & 0xF0) >> 4);
                }
            }
        }

        // release the input frame and mark it as empty
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id++;

    } // end of thread infinite loop

    // Be a nice thread: loop over all the open output buffer frames and release them
    for (auto it = out_frame_map.begin(); it != out_frame_map.end();) {
        // release the output frame in the map
        mark_frame_full(out_buf, unique_name.c_str(), std::get<2>(it->second));
        // retrieve the metadata of the released output buffer frame
        ometadata = std::get<1>(it->second);
        // print out some statistics about the released output buffer frame
        INFO("Released output frame RA: {:f}, Dec: {:f}, scaling: {:d}, freq bin: [{:d},{:d}], "
             "num_freqs: {:d}\n",
             ometadata->ra, ometadata->dec, ometadata->scaling,
             ometadata->lower_band_received_frequency, ometadata->upper_band_received_frequency,
             ometadata->num_received_frequencies);
        // remove the output frame element from the map
        out_frame_map.erase(it++);
    }

    // printout some statistics
    INFO("Beam Assembly statistics for {:d}ms timeout window: \n    number of incoming frames: "
         "{:d}\n"
         "    number of accepted frames: {:d}\n    number of missed frames: {:d}\n",
         arriving_data_timeout, received_beam_frames_count, accepted_beam_frames_count,
         missed_beam_frames_count);
}
