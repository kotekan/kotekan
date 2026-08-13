/**
 * @file
 * @brief Buffers bad input data.
 *  - bufferBadInputs : public kotekan::Stage
 */

#ifndef BUFFER_BAD_INPUT_DATA
#define BUFFER_BAD_INPUT_DATA

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include "json.hpp" // for json

#include <cstdint>  // for int64_t
#include <mutex>    // for lock_guard, mutex
#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class bufferBadInputs
 * @brief CHIME-specific stage which buffers updates to the bad input list.
 *
 * Copies a list of bad inputs into a mask buffer, which is 0 if
 * an element is bad and 1 if it is good.
 *
 * This stage expects the input buffer to be recieved in CHIME cylinder order,
 * and automatically remaps into beamformer order.
 *
 * One frame is produced per output buffer slot, each carrying the mask that is current at
 * that moment, so an update received via the endpoint takes effect on the next frame. Each
 * frame is one bad feed mask sample, and a sample is valid for
 * @c bf_mask_lifetime_in_samples FPGA samples.
 *
 * The mask is a rank-3 array with a leading (length one) time axis, which is what the
 * consumers of the bad feed mask ring buffer expect. Element @c beamformer_idx of the flat
 * CHIME beamformer-order mask is @c polarization * num_dishes + dish (see
 * @c ICETelescope::station_id_to_element_index), so the rank-3 shape is a reinterpretation of
 * the mask rather than a reordering of it.
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer of bad inputs.
 *     @buffer_shape [1, num_polarizations, num_dishes]
 *     @buffer_format int8
 *
 * @conf   num_elements                 Int.  The size of the bad input mask. Must equal
 *                                      num_polarizations * num_dishes.
 * @conf   num_polarizations            Int.  Number of polarizations.
 * @conf   num_dishes                   Int.  Number of dishes.
 * @conf   bf_mask_lifetime_in_samples  Int.  Number of FPGA samples that one bad feed mask
 *                                      is valid for.
 * @conf   updatable_config/bad_inputs  String.  String pointing to the location of the
 *                                      config block containing the following properties:
 *                                      "bad_inputs"  An array of bad inputs in cylinder order.
 *
 * @author James Willis & Liam Gray
 */

class bufferBadInputs : public kotekan::Stage {
public:
    /// Constructor.
    bufferBadInputs(kotekan::Config& config_, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~bufferBadInputs();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;

    /// Endpoint for providing new bad input updates
    bool update_bad_inputs_callback(nlohmann::json& json);

private:
    /// Lock for changing the state of the bad input mask
    std::mutex mtx;

    Buffer* out_buf;
    /// List of current bad inputs in received order, which
    // is expected to be cylinder order
    std::vector<int> bad_inputs;
    /// The size of the bad input mask.
    size_t num_elements;
    /// The shape of the bad input mask, num_elements == num_polarizations * num_dishes
    int num_polarizations;
    int num_dishes;
    /// Number of FPGA samples that one bad feed mask is valid for
    std::int64_t bf_mask_lifetime_in_samples;
    // Number of bad inputs
    uint64_t num_bad_inputs;

    // Store the fixed mask in a permanent buffer
    std::vector<uint8_t> input_mask;

    // The table to reorder from beamformer to cylinder order.
    // reorder[beamformer_idx] = cylinder_idx;
    std::vector<size_t> reorder;
};

#endif
