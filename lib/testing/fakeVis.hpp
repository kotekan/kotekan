/*****************************************
@file
@brief Generate fake visBuffer data.
- fakeVis : public KotekanProcess
*****************************************/

#ifndef FAKE_VIS
#define FAKE_VIS

#include <unistd.h>
#include <random>
#include <string>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"
#include "visBuffer.hpp"

/**
 * @brief Generate fake visibility data into a ``visBuffer``.
 *
 * This process produces fake visibility data that can be used to feed
 * downstream kotekan processes for testing. It fills its buffer with frames in
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
 * @conf  vis_mean_real     When used with mode="gaussian", the real part of
 *                          the mean of the distribution.
 * @conf  vis_mean_imag     When used with mode="gaussian", the imaginary part
 *                          of the mean of the distribution.
 * @conf  vis_std           When used with mode="gaussian", the std dev of the
 *                          distribution.
 * @conf  wait              Bool. Sleep to try and output data at roughly
 *                          the correct cadence.
 * @conf  num_frames        Exit after num_frames have been produced. If
 *                          less than zero, no limit is applied. Default is `-1`.
 * @conf  zero_weight       Bool. Set all weights to zero, if this is True.
 *                          Default is False.
 *
 * @todo  It might be useful eventually to produce realistic looking mock
 *        visibilities.
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class fakeVis : public KotekanProcess {

public:
    /// Constructor. Loads config options.
    fakeVis(Config &config,
            const string& unique_name,
            bufferContainer &buffer_container);

    /// Not yet implemented, should update runtime parameters.
    void apply_config(uint64_t fpga_seq);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread();

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
     * @brief Fill with Gaussian numbers.
     *
     * Fill real and imaginary parts with normally distributed random numbers.
     * Specify mean and standard deviation with additional parameters. Will use
     * the same distribution to set the weights. Note that the seed for the
     * generator is not random.
     *
     * @param frame Frame to fill.
     **/
    void fill_mode_gaussian(visFrameView& frame);

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

private:
    /// Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    /// Output buffer
    Buffer * out_buf;

    /// List of frequencies for this buffer
    std::vector<uint32_t> freq;

    /// Cadence to simulate (in seconds)
    float cadence;

    // Visibility filling mode
    std::string mode;

    // for gaussian modes
    // config values
    cfloat vis_mean;
    float vis_std;
    // random number generation
    std::default_random_engine gen;

    // Test mode that sets all weights to zero
    bool zero_weight;

    bool use_dataset_manager;

    bool wait;
    int32_t num_frames;

    // Alias for the type of a function that will fill a frame.
    using fill_func = std::function<void(visFrameView& frame)>;

    // Mapping of name to type of fill
    std::map<std::string, fill_func> fill_map;
    fill_func fill;

    /// Fill non vis components. A helper for the fill_mode functions.
    void fill_non_vis(visFrameView& frame);
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
class replaceVis : public KotekanProcess {

public:
    /// Constructor. Loads config options.
    replaceVis(Config& config,
               const string& unique_name,
               bufferContainer& buffer_container);

    /// Not yet implemented, should update runtime parameters.
    void apply_config(uint64_t fpga_seq);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread();

private:
    /// Buffers
    Buffer * in_buf;
    Buffer * out_buf;

};
#endif
