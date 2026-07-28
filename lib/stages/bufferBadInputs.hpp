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

#include <mutex>    // for lock_guard, mutex
#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class bufferBadInputs
 * @brief Stage which buffers updates to the bad input list.
 *
 * Copies a list of bad inputs into a mask buffer, which is 0 if
 * an element is bad and 1 if it is good.
 *
 * The incoming bad_inputs indices are element indices in @c input_order; the
 * mask is produced in @c output_order. The defaults preserve the original
 * CHIME behaviour (flags in cylinder order, mask in beamformer order); CHORD
 * sends and masks in the same [P][D] order, so both are CHORDBeamformer and
 * no remapping happens.
 *
 * On a CHORD telescope, elements whose dish type in the dish_inputs table is
 * not ArrayDish (Fake placeholders and RFI antennas) are always masked,
 * independent of the posted bad-inputs list — the behaviour inventBFMask
 * (mode: auto) provided before this stage replaced it. On other telescopes
 * every element starts good.
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer of bad inputs.
 *     @buffer_shape [num_element]
 *     @buffer_format uint8_t
 *
 * @conf   updatable_config/bad_inputs  String.  String pointing to the location of the
 *                                      config block containing the following properties:
 *                                      "bad_inputs"  An array of bad inputs in @c input_order.
 * @conf   input_order   ElementOrder. Default CHIMECylinder.  Order of the incoming
 *                       bad_inputs indices.
 * @conf   output_order  ElementOrder. Default CHIMEBeamformer.  Order of the produced mask.
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
    /// List of current bad inputs in received order (input_order)
    std::vector<int> bad_inputs;
    /// The size of the bad input mask.
    size_t num_elements;
    // Number of bad inputs currently masked
    uint64_t num_bad_inputs = 0;

    // Store the fixed mask in a permanent buffer
    std::vector<uint8_t> input_mask;

    /// Mask before any posted flags: 0 for elements whose CHORD dish is not
    /// an ArrayDish (Fake, RFI antennas); all-1 on other telescopes.
    std::vector<uint8_t> baseline_mask;

    // The table to reorder from output_order to input_order.
    // reorder[output_idx] = input_idx;
    std::vector<size_t> reorder;
};

#endif
