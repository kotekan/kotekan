
/*****************************************
@file
@brief Accumulation and gating of visibility data.
- visAccumulate : public KotekanProcess
*****************************************/
#ifndef VIS_ACCUMULATE_HPP
#define VIS_ACCUMULATE_HPP

#include <vector>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <time.h>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "gateSpec.hpp"
#include "visUtil.hpp"

/**
 * @class visAccumulate
 * @brief Accumulate the high rate GPU output into integrated visBuffers.
 *
 * This process will accumulate the GPU output and calculate the within sample
 * variance for weights.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf
 *         @buffer_format visBuffer
 *         @buffer_metadata visMetadata
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
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  input_reorder         Array of [int, int, string]. The reordering mapping.
 *                              Only the first element of each sub-array is used and it is the the index of
 *                              the input to move into this new location. The remaining elements of the
 *                              subarray are for correctly labelling the input in ``visWriter``.
 * @conf  low_sample_fraction   If a frames has less than this fraction of the
 *                              data expected, skip it. This is set to 1% by default.
 *
 * @par Metrics
 * @metric  kotekan_vis_accumulate_skipped_frame_total
 *      The number of frames skipped entirely because they were under the
 *      low_sample_fraction.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte
 */
class visAccumulate : public KotekanProcess {
public:
    visAccumulate(Config& config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~visAccumulate();
    void apply_config(uint64_t fpga_seq) override;
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
         * @param out_buf   Buffer we will output into.
         * @param gate_spec Specification of how any gating is done.
         **/
        internalState(Buffer* out_buf, std::unique_ptr<gateSpec> gate_spec, size_t nprod);

        /// The buffer we are outputting too
        Buffer* buf;

        // Current frame ID of the buffer we are using
        frameID frame_id;

        /// Specification of how we are gating
        std::unique_ptr<gateSpec> spec;

        /// The weighted number of total samples accumulated. Must be reset every
        /// integration period.
        float sample_weight_total;

        /// Function for applying the weighting. While this can essentially be
        /// derived from the gateSpec we need to cache it so the gating can be
        /// updated externally within an accumulation.
        std::function<float(timespec, timespec, float)> calculate_weight;

        /// Mutex to control update of gateSpec
        std::mutex state_mtx;

        /// Accumulation vectors
        std::vector<cfloat> vis1;
        std::vector<float> vis2;

        friend visAccumulate;
    };

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;  // Output for the main vis dataset only

    // Parameters saved from the config files
    size_t num_elements;
    size_t num_freq_in_frame;
    size_t num_eigenvectors;
    size_t block_size;
    size_t samples_per_data_set;
    size_t num_gpu_frames;
    float low_sample_fraction;

    // Derived from config
    size_t num_prod_gpu;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // Helper methods to make code clearer

    // Combine the gated dataset with the vis dataset to subtract out the bias
    // and have a scaled variance estimate.
    void combine_gated(internalState& gate, internalState& vis);

    // Set initial values of visBuffer
    void initialise_output(internalState& state,
                           int in_frame_id, int freq_ind);

    // Fill in data sections of visBuffer
    void finalise_output(internalState& state, int freq_ind,
                         uint32_t total_samples);

    // List of gating specifications
    std::map<std::string, gateSpec*> gating_specs;

    /**
     * @brief Reset the state when we restart an integration.
     *
     * @returns Return if this accumulation was enabled.
     **/
    bool reset_state(internalState& state);


    // Hold the state for any gated data
    std::deque<internalState> gated_datasets;
};

#endif
