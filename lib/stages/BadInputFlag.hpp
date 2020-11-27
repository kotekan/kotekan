/*****************************************
@file
@brief Check a frame for bad inputs.
- BadInputFlag : public kotekan::Stage
*****************************************/
#ifndef BAD_INPUT_FLAG_HPP
#define BAD_INPUT_FLAG_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily

#include <set>    // for set
#include <string> // for string


/**
 * @class BadInputFlag
 * @brief searches for bad inputs that aren't marked in the `flags` field.
 *
 * @par Buffers
 * @buffer in_buf The buffer from which the visibilities are checked, must be a full triangle.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 * @buffer out_buf The kotekan buffer which will be fed the output.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @par Metrics
 * @metric kotekan_badinputflag_frames_total
 *      The number of frames found with unmarked bad inputs. The metrics are labelled with
 *      which `input` is bad, and why (Infinite weight, NaN weight).
 *
 * @warning  This will only work correctly if the full correlation triangle is
 *           passed in as input.
 *
 * @author  Richard Shaw
 *
 */
class BadInputFlag : public kotekan::Stage {

public:
    /// Constructor. Loads config options. Defines subset of products.
    BadInputFlag(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Primary loop: sorts products and passes them on to output buffer.
    void main_thread() override;

private:
    /// keeps track of the input dataset ID and ensures the products haven't changed
    void check_dataset_state(dset_id_t ds_id);

    /// Input buffer
    Buffer* in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    // Maps for determining the dataset ID to use
    std::set<dset_id_t> dset_id_set;

    fingerprint_t fingerprint = fingerprint_t::null;

    // Count how often a bad input has been seen
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& bad_input_counter;
};


#endif
