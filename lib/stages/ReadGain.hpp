/**
 * @file
 * @brief read in new gain file for FRB/Tracking beamformer when available
 *  - ReadGain : public kotekan::Stage
 */

#ifndef READ_GAIN
#define READ_GAIN

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "Telescope.hpp"         // for freq_id_t
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily

#include "json.hpp" // for json

#include <condition_variable> // for condition_variable
#include <mutex>              // for mutex
#include <queue>              // for queue
#include <stdint.h>           // for int32_t, uint8_t, int16_t, uint32_t
#include <string>             // for string
#include <utility>            // for pair
#include <vector>             // for vector

using std::queue;
using std::vector;

/**
 * @class ReadGain
 * @brief read in new gain files for FRB/Tracking beamformer when available
 *
 * @par Buffers
 * @buffer gain_frb_buf Array of gains size 2048*2
 *     @buffer_format Array of @c floats
 *     @buffer_metadata none
 * @buffer gain_tracking_buf Array of gains size 2048*2*nbeam
 *     @buffer_format Array of @c floats
 *     @buffer_metadata none
 * @buffer in_buf Metadata buffer used to read frequency info.
 *     @buffer_format VisBuffer metadata only
 *     @buffer_metadata VisMetadata
 *
 * @conf   in_buf                String. Metadata buffer (to read frequency).
 * @conf   gain_frb_buf          String. Output buffer for FRB gains.
 * @conf   gain_tracking_buf     String. Output buffer for tracking beam gains.
 * @conf   num_elements          Int (default 2048). Number of elements.
 * @conf   num_beams             Int (default 10). Number of pulsar beams.
 * @conf   frb_scaling           Float (default 1.0). Scaling factor on FRB gains.
 * @conf   frb_missing_gains     Float array (default [0.0, 0.0]). Default gain if file missing.
 * @conf   updatable_config/gain_frb      String. ConfigUpdater path for FRB gain dir updates.
 * @conf   updatable_config/gain_tracking String. Base ConfigUpdater path for tracking gain dirs;
 *                                       one path per beam is constructed by appending `/beam_id`.
 *
 * @par Metrics
 * @metric kotekan_gains_last_update_success
 *     Flag with value 1 if the last gains update succeeded, 0 if failed.
 *     Labels: `type` (values: "frb", "pulsar")
 * @metric kotekan_gains_last_update_timestamp
 *     Timestamp of last gains update.
 *     Labels: `type` (values: "frb", "pulsar")
 *
 * The gain path is registered as a subscriber to an updatable config block.
 * For the FRB, it is one directory path: '{"frb_gain_dir":"the_new_path"}'
 * For the tracking beamformer, it is an array of @c num_beams paths for each of the @c num_beams
 * beams:
 * '{"pulsar_gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'
 *
 * @par Updatable config
 * @updatable_path updatable_config/gain_frb      POST body: {"frb_gain_dir": "/path/to/gains"}
 * @updatable_path updatable_config/gain_tracking/<beam> POST body: {"gain_dir": "/path/to/gains"}
 *
 * @par Example
 * @code
 * ReadGain:
 *   in_buf: vis_in
 *   gain_frb_buf: gains_frb
 *   gain_tracking_buf: gains_tracking
 *   num_elements: 2048
 *   num_beams: 10
 *   frb_scaling: 1.0
 *   frb_missing_gains: [0.0, 0.0]
 *   updatable_config:
 *     gain_frb: "/frb_gain"
 *     gain_tracking: "/tracking_gain"
 * # POST /frb_gain {"frb_gain_dir": "/data/gains/frb"}
 * # POST /tracking_gain/0 {"gain_dir": "/data/gains/tracking_beam0"}
 * @endcode
 *
 * @author Cherry Ng
 *
 */
class ReadGain : public kotekan::Stage {
public:
    /// Constructor.
    ReadGain(kotekan::Config& config_, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    /// Destructor.
    ~ReadGain();

    void main_thread() override;

    /// Endpoint for providing new directory path for FRB gain updates
    bool update_gains_frb_callback(nlohmann::json& json);

    /// Endpoint for providing new directory path for <span class="x x-first x-last">tracking
    /// beamformer</span> gain updates
    bool update_gains_tracking_callback(nlohmann::json& json, const uint8_t beam_id);

    /// Read gain file for frb
    void read_gain_frb();
    /// Read gain file for tracking beamformer
    void read_gain_tracking();

private:
    std::condition_variable cond_var;
    /// Lock when change status of update_gains
    std::mutex mux;

    Buffer* gain_frb_buf;
    int32_t gain_frb_buf_id;
    Buffer* gain_tracking_buf;
    int32_t gain_tracking_buf_id;

    /// Directory path where gain files are
    std::string _gain_dir_frb;
    queue<std::pair<uint8_t, std::string>> _gain_dir_tracking;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    vector<float> default_gains;

    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;

    /// Freq bin index, where the 0th is at 800MHz
    freq_id_t freq_idx;
    /// Freq in MHz
    float freq_MHz;

    /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    /// Flag to control gains to be only loaded on request.
    bool update_gains_frb;
    bool update_gains_tracking;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of pulsar beams, should be 10
    int16_t _num_beams;

    /// Array containing all the current tracking beam gains
    float* tracking_beam_gains;

    /// implements `kotekan_gains_last_update_success`
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& gains_last_update_success_metric;

    /// implements `kotekan_gains_last_update_timestamp`
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        gains_last_update_timestamp_metric;
};


#endif
