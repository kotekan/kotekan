/*****************************************
@file
@brief Generate fake hfbBuffer data.
- FakeHFB : public Stage
- ReplaceHFB : public Stage
*****************************************/

#ifndef FAKE_HFB
#define FAKE_HFB

#include "Config.hpp"          // for Config
#include "HFBFrameView.hpp"    // for HFBFrameView
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "dataset.hpp"         // for dset_id_t
#include "visUtil.hpp"         // for cfloat

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Generate fake absorber data into a ``hfbBuffer``.
 *
 * This stage produces fake absorber data that can be used to feed
 * downstream kotekan stages for testing. It fills its buffer with frames in
 * the ``HFBFrameView`` format. Frames are generated for a set of frequencies
 * and a cadence specified in the config.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed, can be any size.
 *     @buffer_format hfbBuffer structured
 *     @buffer_metadata HFBMetadata
 *
 * @conf  num_beams     Int. The number of beams in the absorber data.
 * @conf  num_subfreq   Int. The number of sub-frequencies in the absorber data..
 * @conf  freq_ids      List of int. The frequency IDs to generate frames
 *                      for.
 * @conf  start_time    Double. The start time of the range of data (as a
 *                      Unix time in seconds). This simply changes the time
 *                      the frames are labelled with. Default is the current
 *                      time.
 * @conf  cadence       Float. The interval of time (in seconds) between
 *                      frames.
 * @conf  mode          String. How to fill the absorber array. See
 *                      the set of FakeHFBPattern implementations for details.
 * @conf  wait          Bool. Sleep to try and output data at roughly
 *                      the correct cadence.
 * @conf  num_frames    Exit after num_frames have been produced. If
 *                      less than zero, no limit is applied. Default is `-1`.
 * @conf  zero_weight   Bool. Set all weights to zero, if this is True.
 *                      Default is False.
 * @conf  freq          Array of UInt32. Definition of frequency IDs for
 *                      mode 'test_pattern_freq'.
 * @conf  sleep_before  Float. Sleep for this number of seconds before
 *                      starting. Useful for allowing other processes
 *                      to send REST commands. Default is 0s.
 * @conf  sleep_after   Float. Sleep for this number of seconds after running
 *                      and before shutting down. Useful for allowing other
 *                      processes to finish. Default is 1s.
 *
 * @author  James Willis
 *
 */
class FakeHFB : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    FakeHFB(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:
    /// Parameters saved from the config files
    size_t num_beams, num_subfreq;

    /// Config parameters for freq or inputs test pattern
    std::vector<cfloat> test_pattern_value;

    /// Output buffer
    Buffer* out_buf;

    /// List of frequencies for this buffer
    std::vector<uint32_t> freq;

    /// Test pattern
    // std::unique_ptr<FakeHFBPattern> pattern;

    /// Start time of data as a Unix time
    double start_time;

    /// Cadence to simulate (in seconds)
    float cadence;

    // Visibility filling mode
    std::string mode;

    // Test mode that sets all weights to zero
    bool zero_weight;

    bool wait;
    int32_t num_frames;

    // How long to sleep before starting.
    double sleep_before;

    // How long to sleep before exiting.
    double sleep_after;

    /// Fill non hfb components. A helper for the fill_mode functions.
    void fill_non_hfb(HFBFrameView& frame);

    // Use a fixed (configured) dataset ID in the output frames
    bool _fixed_dset_id;
    dset_id_t _dset_id;
};


/**
 * @brief Copy a buffer and replace its data with test data.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer which will be read from.
 *     @buffer_format hfbBuffer structured
 *     @buffer_metadata HFBMetadata
 * @buffer out_buf The kotekan buffer to be filled with the replaced data.
 *     @buffer_format hfbBuffer structured
 *     @buffer_metadata HFBMetadata
 *
 * @author Richard Shaw
 *
 */
class ReplaceHFB : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    ReplaceHFB(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:
    /// Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif
