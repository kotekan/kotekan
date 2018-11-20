/*****************************************
@file
@brief Reduce cadence of a single-frequency.
- timeDownsample : public KotekanProcess
*****************************************/
#ifndef TIME_DOWNSAMP_HPP
#define TIME_DOWNSAMP_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

/**
 * @class timeDownsample
 * @brief Average a set number of frames on a single-frequency stream to effectively
 *        reduce the cadence of the acquisition.
 *
 * This process accumulates and averages a specified number of incoming frames on a
 * single-frequency stream to reduce the cadence of the acquisition. Visibilities,
 * eigenvectors, eigenvalues, eigen-rms are averaged. Inverse weights are averaged
 * and divided by number of combined frames to track reduction in variance. Metadata
 * from the first frame is passed on and that of the others discarded.
 * Will throw an exception if more than one frequency is found in the stream.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer of the incoming single-frequency stream.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer into which low cadence stream is fed.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 *
 * @conf  num_samples       Int. The number of time frames to average.
 *
 * @warning There is not mechanism in place to verify or enforce that frames being
 *          averaged are concurrent in time. Since this process is run for a single
 *          frequency stream, there is no guarantee that independent streams will have
 *          consistent timestamps.
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class timeDownsample : public KotekanProcess {

public:

    /// Default constructor, loads config params.
    timeDownsample(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container);

    /// Main loop for the process
    void main_thread() override;

private:

    // Frame parameters
    size_t num_elements, num_eigenvectors;
    size_t nprod;

    // Number of samples to combine
    int nsamp;

    // Buffers
    Buffer * in_buf;
    Buffer * out_buf;

};

#endif
