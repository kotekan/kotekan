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

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class bufferBadInputs
 * @brief Buffers updates to the bad input list.
 *
 * This engine reorders, inverts and generates a mask of bad inputs then stores the mask in a buffer
 * whenever an update arrives via configUpdater.
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer of bad inputs.
 *     @buffer_format Array of @c uint8_t, length = num_elements, 1 = good, 0 = bad
 *     @buffer_metadata chordMetadata (used to set @c rfi_num_bad_inputs)
 *
 * @conf   num_elements                Int. Number of inputs (length of mask).
 * @conf   input_reorder               Array. Optional cylinder->correlator mapping
 *                                     (see @ref visUtil::parse_reorder_default).
 * @conf   updatable_config/bad_inputs String. Path for configUpdater updates containing
 *                                     {"bad_inputs":[...]} in cylinder order.
 *
 * @par REST/Updatable config
 * @updatable_path updatable_config/bad_inputs
 *
 * @par Example
 * @code
 * bufferBadInputs:
 *   num_elements: 2048
 *   input_reorder: [[0]]
 *   updatable_config:
 *     bad_inputs: "/updatable_config/bad_inputs"
 *   out_buf: bad_inputs_buf
 * @endcode
 *
 * @author James Willis
 *
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
    Buffer* out_buf;

    /// Stage variables

    /// List of current bad inputs in cylinder order
    std::vector<int> bad_inputs_cylinder;

    /// List of current bad inputs in correlator order.
    std::vector<int> bad_inputs_correlator;

    /// The size of the bad input mask.
    uint32_t input_mask_len;

    /// The mapping from correlator to cylinder element indexing.
    std::vector<uint32_t> input_remap;

    uint32_t out_buffer_ID = 0;
};

#endif
