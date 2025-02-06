/*****************************************
@file
@brief Reduce cadence of a single-frequency.
- N2FringeStop : public kotekan::Stage
*****************************************/
#ifndef N2_FRINGE_STOP_HPP
#define N2_FRINGE_STOP_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <stddef.h> // for size_t
#include <string>   // for string

/**
 * @class N2FringeStop
 * @brief Average a set number of frames on a single-frequency stream to
 *        effectively reduce the cadence of the acquisition.
 *
 * This stage accumulates and averages a specified number of incoming frames on 
 * a single-frequency stream to reduce the cadence of the acquisition.
 * Visibilities, eigenvectors, eigenvalues, eigen-rms are averaged. Inverse
 * weights are averaged and divided by number of combined frames to track
 * reduction in variance. Metadata from the first frame is passed on and that
 * of the others discarded.
 * Will throw an exception if more than one frequency is found in the stream.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer of the incoming single-frequency stream.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 * @buffer out_buf The kotekan buffer into which low cadence stream is fed.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @conf  num_samples  Int. The number of time frames to average.
 * @conf  max_age      Float. How old can a frame be in seconds before it is dropped.
 *                     Default is 120 (i.e. two minutes).
 *
 * @metric  kotekan_timedownsample_skipped_frame_total
 *      The number of frames skipped entirely because they were too old.
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class N2FringeStop : public kotekan::Stage {

public:
    /// Default constructor, loads config params.
    N2FringeStop(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Main loop for the stage
    void main_thread() override;

private:
    // Frame parameters
    size_t num_elements, num_eigenvectors;
    size_t nprod;

    // Whether to apply fringestopping phases.
    int fringestop_mode;

    double era_target_deg;
    double xp_target_as;
    double yp_target_as;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif
