// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
* @file   BeamAssemble.hpp
* @brief  This file declares the BeamAssemble class which
*         takes an input buffer with frames for single 
*         frequency, and assembles the frames for a given 
*         number of frequencies in a larger buffer.
*
* @author Mehdi Najafi
* @date   28 AUG 2022
*****************************************************/

#ifndef KOTEKAN_BEAMASSEMBLE_HPP
#define KOTEKAN_BEAMASSEMBLE_HPP

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "bufferContainer.hpp"  // for bufferContainer

/**
 * @class BeamAssemble
 * @brief Assembles incoming beam information from the frames in 
 *        the in_buf and gather in a larger format to an out_buf.
 * *
 * @par Buffers
 * @buffer in_buf The buffer to receive data from.
 *     @buffer_format 4+4-bit complex voltage (beam) data
 *     @buffer_metadata BeamMetadata
 * @buffer out_buf The buffer to put the data to.
 *     @buffer_format 4+4-bit complex voltage (beam) data
 *     @buffer_metadata BeamMetadata
 * 
 * @conf num_freq_per_output_frame
 *      number of frequencies to be assembled
 * 
 * @conf arriving_data_timeout
 *      timeout in miliseconds for the arriving data
 * 
 * @conf beam_printout
 *      check flag to print some information about incoming beam 
 *      data
 * 
 * @conf late_beam_printout
 *      check flag to print some information about late incoming 
 *      beam data, which are ignored
 * 
 * @par received_beam_frames_count
 *      a counter to keep track of the number of beam frames received
 * 
 * @par accepted_beam_frames_count
 *      a counter to keep track of the number of beam frames received
 *      and copied to the output buffer frames
 *
 * @par missed_beam_frames_count
 *      a counter to keep track of the number of beam frames received
 *      late and ignored (not copied to any output buffer frames)
 *
 * @author Mehdi Najafi
 */
class BeamAssemble : public kotekan::Stage {
public:
    BeamAssemble(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~BeamAssemble();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    struct Buffer* out_buf;
    uint32_t num_freq_per_output_frame;
    int32_t arriving_data_timeout;

    bool beam_printout;
    bool late_beam_printout;

    uint64_t received_beam_frames_count;
    uint64_t accepted_beam_frames_count;
    uint64_t missed_beam_frames_count;
};

#endif // KOTEKAN_BEAMASSEMBLE_HPP