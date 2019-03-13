/*****************************************
@file
@brief Generate fake visBuffer data.
- fakeVis : public Stage
*****************************************/

#ifndef FAKE_VIS
#define FAKE_VIS

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include <functional>
#include <map>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>


/**
 * @brief Generate fake visibility data into a ``visBuffer``.
 *
 * This stage produces fake visibility data that can be used to feed
 * downstream kotekan stages for testing. It fills its buffer with frames in
 * the ``visFrameView`` format. Frames are generated for a set of frequencies
 * and a cadence specified in the config.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed, can be any size.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 *
 * @conf  num_elements      Int. The number of elements (i.e. inputs) in the
 *                          correlator data,
 * @conf  block_size        Int. The block size of the packed data.
 * @conf  num_ev            Int. The number of eigenvectors to be stored.
 * @conf  freq_ids          List of int. The frequency IDs to generate frames
 *                          for.
 * @conf  cadence           Float. The interval of time (in seconds) between
 *                          frames.
 * @conf  mode              String. How to fill the visibility array. See
 *                          fakeVis::fill_mode_X routines for documentation.
 * @conf  wait              Bool. Sleep to try and output data at roughly
 *                          the correct cadence.
 * @conf  num_frames        Exit after num_frames have been produced. If
 *                          less than zero, no limit is applied. Default is `-1`.
 * @conf  zero_weight       Bool. Set all weights to zero, if this is True.
 *                          Default is False.
 * @conf  default_val       Cfloat. The default test pattern value for modes
 *                          'test_pattern_simple' and 'test_pattern_freq'.
 * @conf  frequencies       Array of UInt32. Definition of frequency IDs for
 *                          mode 'test_pattern_freq'.
 * @conf  freq_values       Array of CFloat. Values for the frequency IDs in
 *                          mode 'test_pattern_freq'.
 * @conf  dataset_id        Int. Use a fixed dataset ID and don't register
 *                          states. If not set, the dataset manager will create
 *                          the dataset ID.
 *
 * @todo  It might be useful eventually to produce realistic looking mock
 *        visibilities.
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class fakeVis : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    fakeVis(kotekan::Config& config, const string& unique_name,
            kotekan::bufferContainer& buffer_container);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

    /**
     * @brief Default fill pattern.
     *
     * The visibility array is populated with integers increasing from zero on
     * the diagonal (imaginary part) and FPGA sequence number, timestamp,
     * frequency, and frame ID in the first four elements (real part). The
     * remaining elements are zero.
     *
     * @param frame Frame to fill.
     **/
    void fill_mode_default(visFrameView& frame);

    /**
     * @brief Default fill pattern.
     *
     * Fill the real part with the index of feed i and the imaginary part with
     * the index of j.
     *
     * @param frame Frame to fill.
     **/
    void fill_mode_fill_ij(visFrameView& frame);

    /**
     * @brief Fill with a factorisable pattern.
     *
     * Fill with unit amplitude numbers with phase ``i - j``
     * radians.
     *
     * @param frame Frame to fill.
     **/
    void fill_mode_phase_ij(visFrameView& frame);

    /**
     * @brief Fill with a pattern to test CHIME redundant stacking.
     *
     * Fill real and imaginary parts with normally distributed random numbers.
     * Specify mean and standard deviation with additional parameters. Will use
     * the same distribution to set the weights. Note that the seed for the
     * generator is not random.
     *
     * @param frame Frame to fill.
     **/
    void fill_mode_chime(visFrameView& frame);

    /**
     * @brief Fill with a simple test pattern, where all visibilities have
     * the value of 'default_val'.
     *
     * @param frame Frame to fill.
     */
    void fill_mode_test_pattern_simple(visFrameView& frame);

    /**
     * @brief Fill with a frequency dependend test pattern, where the
     * frequencies defined in the config value 'frequencies' have the values
     * defined in 'freq_values'. All other visibility values have the value
     * defined in 'default_val'.
     *
     * @param frame Frame to fill.
     */
    void fill_mode_test_pattern_freq(visFrameView& frame);

    /**
     * @brief Fill with a input dependend test pattern, where the
     * input values are defined in the config value 'input_values'.
     *
     * @param frame Frame to fill.
     */
    void fill_mode_test_pattern_inputs(visFrameView& frame);

private:
    /// Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    /// Config parameters for freq or inputs test pattern
    std::vector<cfloat> test_pattern_value;

    /// Output buffer
    Buffer* out_buf;

    /// List of frequencies for this buffer
    std::vector<uint32_t> freq;

    /// Cadence to simulate (in seconds)
    float cadence;

    // Visibility filling mode
    std::string mode;

    // Test mode that sets all weights to zero
    bool zero_weight;

    bool wait;
    int32_t num_frames;

    // Alias for the type of a function that will fill a frame.
    using fill_func = std::function<void(visFrameView& frame)>;

    // Mapping of name to type of fill
    std::map<std::string, fill_func> fill_map;
    fill_func fill;

    /// Fill non vis components. A helper for the fill_mode functions.
    void fill_non_vis(visFrameView& frame);

    // Use a fixed (configured) dataset ID in the output frames
    bool _fixed_dset_id;
    dset_id_t _dset_id;
};


/**
 * @class replaceVis
 * @brief Copy a buffer and replace its data with test data.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer which will be read from.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer to be filled with the replaced data.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 *
 * @author Richard Shaw
 *
 */
class replaceVis : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    replaceVis(kotekan::Config& config, const string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:
    /// Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};
#endif
