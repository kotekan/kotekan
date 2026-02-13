/*****************************************
@file
@brief Miscellaneous utils for the trackinb beam recording code.
- Merge the correlator tracking beam frames to one frame. 
*****************************************/
#ifndef BEAM_UTIL_HPP
#define BEAM_UTIL_HPP


#include "Config.hpp" // for Config
#include "buffer.h"   // for Buffer



/**
 * @brief Add one single burst data frame to a bigger frame that has mulitpy frames.
 * @param inputframe    Input single frame 
 * @param start_pos     Start position where the single burst frame gets copied to.
 * @param inframe_size  Size of the input frame size.
 * @param N         Number of inputs in input data.
 * @param output    Region of memory to write into.
 */

void merge_frames(uint8_t* inputframe);
