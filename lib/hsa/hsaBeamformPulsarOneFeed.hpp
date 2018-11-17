/*****************************************
File Contents:
- hsaBeamformPulsarOneFeed : public hsaCommand
*****************************************/

#ifndef HSA_BEAMFORM_PULSAR_ONE_FEED_H
#define HSA_BEAMFORM_PULSAR_ONE_FEED_H

#include "hsaCommand.hpp"

/**
 * @class hsaBeamformPulsarOneFeed
 * @brief hsaCommand to zero-out all but one feed
 *
 *
 * This is an interim code that pretends to form pulsar beams. 
 * Because we don't have proper gain calibration at the moment, it doesn't make 
 * sense to phase up the feeds. Instead, this code generates an array of phases 
 * that effectively zeros out all but one feed. It still goes through the 
 * same amount of computations as if it were to form the beams properly so 
 * this code might also give an indication of the performance. All x (default 10) 
 * output beams are supposed to be identical. 
 * 
 *
 * @buffer  input           Input data of size input_frame_len
 *     @buffer_format Array of @c uchar
 *     @buffer_metadata yes
 * @buffer  bf_output       Output data of size output_frame_len
 *     @buffer_format Array of @c uint8_t
 *     @buffer_metadata yes
 * @buffer  beamform_phase  Array of phases of size phase_len
 *     @buffer_format Array of @c float
 *     @buffer_metadata none
 *
 *
 * @conf   _num_elements         Int (default 2048). Number of elements
 * @conf   _num_pulsar           Int (default 10). Number of pulsar beams to be formed
 * @conf   _samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   _num_pol              Int (default 2). Number of polarizations
 * @conf   _one_feed_p0          Int. The index of the desired feed (pol 0)
 * @conf   _one_feed_p1          Int. The index of the desired feed (pol 1)
 * @conf   command               String (defualt: ""). Kernel command.
 * @conf   kernel                String (default: ""). Kernel filename.
 *
 * @remark The two indices of feed_p0 and feed_p1 could in principle come from 
 *         two different feeds. But since we can't calibrate at the moment, it  
 *         probably makes the most sense to have them being the two pol of the 
 *         same feed.
 *
 * @author Cherry Ng
 *
 */

class hsaBeamformPulsarOneFeed: public hsaCommand
{
public:
    /// Constructor, also initializes internal variables from config and generate the array of phases.
    hsaBeamformPulsarOneFeed(Config &config, const string &unique_name,
                        bufferContainer &host_buffers, hsaDeviceInterface &device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaBeamformPulsarOneFeed();

    /// Allocate kernetl argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id,
                         hsa_signal_t precede_signal) override;

private:
    // Input length, should be nsamp x n_elem x 2 for complex / 2 since we pack two 4-bit in one
    int32_t input_frame_len;
    // Output length, should be 10psr x nsamp x 2 pol x 2 for complex / 2 since we pack two 4-bit in one
    int32_t output_frame_len;


    //Length of the array of phases for beamforming, should be 10 psr * 2048 elem * 2 for complex
    int32_t phase_len;
    // pointer to the phase array
    float * host_phase;


    //numbler of elements, should be 2048 
    int32_t _num_elements;
    // number of pulsar beams to be formed, should be 10 
    int32_t _num_pulsar;
    // number of polarizations in the data, should be 2
    int32_t _num_pol;
    // index of the required feed (pol 0), scrambled (correlator based) order
    int32_t _one_feed_p0;
    // index of the required feed (pol 1), scrambled (correlator based) order
    int32_t _one_feed_p1;
    // number of samples
    int32_t _samples_per_data_set;
};

#endif
