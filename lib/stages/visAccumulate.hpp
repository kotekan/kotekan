
/*****************************************
@file
@brief Accumulation and gating of visibility data.
- visAccumulate : public kotekan::Stage
*****************************************/
#ifndef VIS_ACCUMULATE_HPP
#define VIS_ACCUMULATE_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t
#include "gateSpec.hpp"          // for gateSpec
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, freq_ctype, input_ctype, prod_ctype

#include <cstdint>    // for uint32_t, int32_t
#include <deque>      // for deque
#include <functional> // for function
#include <map>        // for map
#include <memory>     // for unique_ptr
#include <mutex>      // for mutex
#include <string>     // for string
#include <time.h>     // for size_t, timespec
#include <utility>    // for pair
#include <vector>     // for vector


/**
 * @class visAccumulate
 * @brief Accumulate the high rate GPU output into integrated VisBuffers.
 *
 * This stage will accumulate the GPU output and calculate the within sample
 * variance for weights.
 *
 * It tags the stream with a properly allocated dataset_id and adds associated
 * datasetStates to the datasetManager.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf The accumulated and tagged data.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata VisMetadata
 *
 * @conf  samples_per_data_set  Int. The number of samples each GPU buffer has
 *                              been integrated for.
 * @conf  num_gpu_frames        Int. The number of GPU frames to accumulate over.
 * @conf  integration_time      Float. Requested integration time in seconds.
 *                              This can be used as an alterative to
 *                              `num_gpu_frames` (which it overrides).
 *                              Internally it picks the nearest acceptable value
 *                              of `num_gpu_frames`.
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 * @conf  block_size            Int. The block size of the packed data.
 * @conf  input_reorder         Array of [int, int, string]. The reordering mapping.
 *                              Only the first element of each sub-array is used and
 *                              it is the the index of the input to move into this
 *                              new location. The remaining elements of the subarray
 *                              are for correctly labelling the input in
 *                              ``Writer``.
 * @conf  low_sample_fraction   If a frames has less than this fraction of the
 *                              data expected, skip it. This is set to 1% by default.
 * @conf  instrument_name       String. Name of the instrument. Default "chime".
 * @conf  freq_ids              Vector of UInt32. Frequency IDs on the stream.
 *                              Default 0..1023.
 * @conf  max_age               Float. Drop frames later than this number of seconds.
 *                              Default is 60.0
 * @conf  fpga_dataset          String. The dataset ID for the data being received from
 *                              the F-engine.
 *
 * @par Metrics
 * @metric  kotekan_visaccumulate_skipped_frame_total
 *      The number of frames skipped entirely because they were under the
 *      low_sample_fraction, or too old.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte
 */
class visAccumulate : public kotekan::Stage {
public:
    visAccumulate(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~visAccumulate() = default;
    void main_thread() override;

private:
    // NOTE: Annoyingly this can't be forward declared, and defined fully externally
    // as the std::deque needs the complete type
    /**
     * @class internalState
     * @brief Hold the internal state of a gated accumulation.
     **/
    struct internalState {

        /**
         * @brief Initialise the required fields.
         *
         * Everything else will be set by the reset_state call during
         * initialisation.
         *
         * @param  out_buf    Buffer we will output into.
         * @param  gate_spec  Specification of how any gating is done.
         * @param  nprod      Number of products.
         **/
        internalState(Buffer* out_buf, std::unique_ptr<gateSpec> gate_spec, size_t nprod);

        /// View of the data accessed by their freq_ind
        std::vector<VisFrameView> frames;

        /// The buffer we are outputting too
        Buffer* buf;

        // Current frame ID of the buffer we are using
        frameID frame_id;

        /// Specification of how we are gating
        std::unique_ptr<gateSpec> spec;

        /// The weighted number of total samples accumulated. Must be reset every
        /// integration period.
        float sample_weight_total;

        /// The sum of the squared weight difference. This is needed for
        /// de-biasing the weight calculation
        float weight_diff_sum;

        /// Function for applying the weighting. While this can essentially be
        /// derived from the gateSpec we need to cache it so the gating can be
        /// updated externally within an accumulation.
        std::function<float(timespec, timespec, float)> calculate_weight;

        /// Mutex to control update of gateSpec
        /// ... and bool to signal changes (should only be changed when locked)
        std::mutex state_mtx;
        bool changed;

        /// Accumulation vectors
        std::vector<int32_t> vis1;
        std::vector<float> vis2;

        /// Dataset ID for output
        dset_id_t output_dataset_id;

        friend visAccumulate;
    };

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf; // Output for the main vis dataset only

    // Parameters saved from the config files
    size_t num_elements;
    size_t num_freq_in_frame;
    size_t block_size;
    size_t samples_per_data_set;
    size_t num_gpu_frames;
    size_t minimum_samples;
    float max_age;
    dset_id_t fpga_dataset;

    // Derived from config
    size_t num_prod_gpu;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // Helper methods to make code clearer

    /**
     * @brief Construct the correct gated visibilities from the gated and
     *        ungated dataset.
     *
     * @param  gate  The gated dataset.
     * @param  vis   The ungated dataset.
     **/
    void combine_gated(internalState& gate, internalState& vis);

    /**
     * @brief Allocate the frame and initialise the VisBuffer's for each freq.
     *
     * This routine will wait on an empty frame to become available on the output buffer.
     *
     * @param  state        The dataset to proces.
     * @param  in_frame_id  The position of the input frame. Needed to get the
     *                      metadata.
     * @returns             True if kotekan was stopped while waiting for the
     *                      buffer.
     **/
    bool initialise_output(internalState& state, int in_frame_id);

    /**
     * @brief Fill in the data sections of VisBuffer and release the frame.
     *
     * @param  state              Dataset to process.
     * @param  newest_frame_time  Used for deciding how late a frame is. A UNIX
     *                            time in seconds.
     **/
    void finalise_output(internalState& state, timespec newest_frame_time);

    /**
     * @brief Reset the state when we restart an integration.
     *
     * @param    state  State to reset.
     * @param    t      Current time.
     * @returns         True if this accumulation was enabled.
     **/
    bool reset_state(internalState& state, timespec t);

    // List of gating specifications
    std::map<std::string, gateSpec*> gating_specs;

    // Hold the state for any gated data
    std::deque<internalState> gated_datasets;

    // dataset ID for the base (input, prod, freq, meta)
    dset_id_t base_dataset_id;


    /// Sets the metadataState with a hardcoded weight type ("inverse_var"),
    /// prodState, inputState and freqState according to config and an empty
    /// stackState
    dset_id_t base_dataset_state(std::string& instrument_name,
                                 std::vector<std::pair<uint32_t, freq_ctype>>& freqs,
                                 std::vector<input_ctype>& inputs, std::vector<prod_ctype>& prods);

    /// Register a new state with the gating params
    dset_id_t gate_dataset_state(const gateSpec& spec);

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
