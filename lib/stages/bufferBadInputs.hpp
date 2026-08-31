/**
 * @file
 * @brief Buffers bad input data.
 *  - bufferBadInputs : public kotekan::Stage
 */

#ifndef BUFFER_BAD_INPUT_DATA
#define BUFFER_BAD_INPUT_DATA

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter
#include "updateQueue.hpp"       // for updateQueue

#include "json.hpp" // for json

#include <atomic>   // for atomic
#include <stddef.h> // for size_t
#include <stdint.h> // for int64_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class bufferBadInputs
 * @brief CHIME-specific stage which buffers updates to the bad input list.
 *
 * Copies a list of bad inputs into a mask buffer, which is 0 if
 * an element is bad and 1 if it is good.
 *
 * The posted indices are in @c input_order and the produced mask is in
 * @c output_order, remapped through the telescope.  The defaults are the
 * CHIME orders; when the two are equal -- CHORD flags and masks in the same
 * [P][D] order -- the telescope is not consulted.
 *
 * Elements whose CHORD dish is not an ArrayDish (Fake dishes, RFI antennas)
 * are never valid inputs; they are masked from the telescope's dish table and
 * stay masked whatever the posted list says.
 *
 * Updates queue by @c start_time and take effect once the wall clock reaches
 * it (a start time already in the past applies immediately).  Mask frames
 * are produced whenever the output buffer has room, so the stage is paced by
 * its consumers.
 *
 * Each frame is one bad feed mask sample, valid
 * for @c bf_mask_lifetime_in_samples FPGA samples, and its FPGA sequence
 * number is that sample's seq -- so it grows by the lifetime from frame to
 * frame, which is what the bad feed mask ring buffer requires.
 *
 * Out-of-order and malformed updates are counted and ignored while running
 * -- a bad POST must not stop the correlator -- but a malformed *initial*
 * config block is still fatal.
 *
 * The mask is a rank-3 array with a leading (length one) time axis, which is
 * what the consumers of the bad feed mask ring buffer expect.  Element
 * @c output_idx of the flat mask is @c polarization * num_dishes + dish in
 * both of the pol-major output orders the stage is used with (see
 * @c ICETelescope::station_id_to_element_index), so the rank-3 shape is a
 * reinterpretation of the mask rather than a reordering of it.
 *
 * @par Buffers
 * @buffer out_buf Kotekan buffer of bad inputs (1 == good).
 *     @buffer_shape [1, num_polarizations, num_dishes]
 *     @buffer_format int8
 *
 * @par Metrics
 * @metric kotekan_bufferbadinputs_late_update_count  Updates ignored because
 *     their start_time preceded an already-queued update.
 * @metric kotekan_bufferbadinputs_invalid_update_count  Updates ignored
 *     because they could not be parsed or held an out-of-range index.
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
 *                                      "start_time"  Optional UNIX time the update takes
 *                                                    effect (default: immediately).
 *                                      "update_id"   Optional string identifying the update,
 *                                                    used in this stage's log messages.
 * @conf   num_kept_updates  Int. Default 5.  Number of updates kept in the queue.
 * @conf   input_order   ElementOrder. Default CHIMECylinder.  Order of the posted
 *                       "bad_inputs" indices.
 * @conf   output_order  ElementOrder. Default CHIMEBeamformer.  Order of the mask
 *                       written to @c out_buf.
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
    /// A posted update, ready to be written into a mask frame.
    struct badInputUpdate {
        /// The mask in output_order, 1 == good.
        std::vector<uint8_t> mask;
    };

    Buffer* out_buf;
    /// The size of the bad input mask.
    size_t num_elements;
    /// The shape of the bad input mask, num_elements == num_polarizations * num_dishes
    int num_polarizations;
    int num_dishes;
    /// Number of FPGA samples that one bad feed mask is valid for
    int64_t bf_mask_lifetime_in_samples;

    /// Mask before any posted flags: 0 for elements whose CHORD dish is not
    /// an ArrayDish (Fake, RFI antennas); all-1 on other telescopes.
    std::vector<uint8_t> baseline_mask;

    /// Posted updates, keyed by their start time.
    updateQueue<badInputUpdate> updates;

    /// False until the constructor's subscribe() has returned, i.e. while the
    /// initial config block is being applied. A bad update is fatal then and
    /// merely counted afterwards.
    std::atomic<bool> initialised = false;

    kotekan::prometheus::Counter& late_updates_counter;
    kotekan::prometheus::Counter& invalid_updates_counter;

    /// The table to reorder from input_order to output_order.
    /// reorder[input_idx] = output_idx;
    std::vector<size_t> reorder;
};

#endif
