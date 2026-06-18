/**
 * @file
 * @brief read in new gain file(s) for FRB/Tracking beamformer when available
 *  - processGains : public kotekan::Stage
 */

#ifndef PROCESS_GAINS_HPP
#define PROCESS_GAINS_HPP

#include "Config.hpp"            // for Config
#include "N2Util.hpp"            // for frameID
#include "Stage.hpp"             // for Stage
#include "Telescope.hpp"         // for freq_id_t
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily

#include "json.hpp" // for json

#include <atomic>   // for atomic_flag
#include <mutex>    // for mutex
#include <queue>    // for queue
#include <stdint.h> // for int32_t, uint8_t, int16_t, uint32_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

using std::queue;
using std::vector;

/**
 * @class processGains
 * @brief Read in new gain files for FRB/Tracking beamformer when available.
 *
 * New gains are indicated by an updatable config endpoint, and the current gains
 * are held in a persistent buffer. Files are loaded per-frequency and merged into
 * a single buffer, with frequency as the outer axis.
 *
 * FRB gains are upchannelized by a factor of `frb_upchan_factor` in frequency. The
 * current implementation does nothing more than copy each coarse frequency gain
 * `frb_upchan_factor` times.
 *
 * In order to be compatible with existing copy-to-GPU methods, this stage will
 * continuously emit kotekan buffers containing the current fan beam gains and
 * tracking beams gains (as two separate buffers).
 *
 * @par Buffers
 * @buffer in_buf Arbitrary buffer used to collect metadata. Must contain _all_
 *                local frequencies.
 *     @buffer_format any
 *     @buffer_shape any
 *     @buffer_metadata chordMetadata
 * @buffer gain_frb_buf Array of gains used by the fan beamformer.
 *     @buffer_format float32
 *     @buffer_shape [freq][upchan_factor][element][real/imag]
 *     @buffer_metadata chordMetadata
 * @buffer gain_tracking_buf Array of per-beam gains used by the tracking beamformer.
 *     @buffer_format float32
 *     @buffer_shape [freq][beam][element][real/imag]
 *     @buffer_metadata chordMetadata
 *
 * @conf   num_elements         Int. Number of elements
 * @conf   num_beams            Int. Number of beams with unique gain files
 * @conf   num_local_freq       Int. Number of frequencies per node
 * @conf   frb_upchan_factor    Int. Frequency upchannelization factor. Only applies
 *                                to FRB gains
 * @conf   scaling              Float (default 1.0). Scaling factor on gains
 * @conf   default_gains        Float array (default 0 + 0j). Default gain value if gain file is
 *                                missing.
 *
 * @par Metrics
 * @metric kotekan_gains_last_update_success
 *     Flag with value 1 if the last gains update succeeded, 0 if failed.
 *     Labels: `type` (values: "frb", "tracking")
 *             `beam_id` (uint32_t)
 * @metric kotekan_gains_last_update_timestamp
 *     Timestamp of last gains update.
 *     Labels: `type` (values: "frb", "tracking")
 *             `beam_id` (uint32_t)
 *
 * The gain path is registered as a subscriber to an updatable config block.
 * For the FRB, the update payload is one directory path: '{"gain_dir":"the_new_path"}'
 * For the tracking beams, updates are provided on per-beam endpoints, each with one directory
 * path for the corresponding beam: `{"gain_dir: "path_for_beam"}`.
 *
 * @author Cherry Ng & Liam Gray
 *
 */
class processGains : public kotekan::Stage {
public:
    /// Constructor.
    processGains(kotekan::Config& config_, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Destructor.
    ~processGains();

    void main_thread() override;

    /// Endpoint for providing new directory path for FRB gain updates
    bool update_gains_frb_callback(nlohmann::json& json);
    /// Endpoint for providing new directory path for tracking beamformer gain updates
    bool update_gains_tracking_callback(nlohmann::json& json, const uint32_t beam_id);

    /// Load new FRB gains from the provided directory and upchannelize
    void update_gain_frb();
    /// Load new tracking beamformer gains from the provided directory
    void update_gain_tracking();

private:
    /// Apply an upchannelization algorithm to the FRB gains
    /// NB: at the moment this just duplicates the gains across all
    /// fine-resolution frequency channels
    void upchannelize_gain_frb();

    /// Lock when change status of update_gains
    std::mutex mux;

    Buffer* gain_frb_buf;
    Buffer* gain_tracking_buf;
    N2::frameID gain_frb_frame_id;
    N2::frameID gain_tracking_frame_id;

    /// Directory path where gain files are
    std::string _gain_dir_frb;
    queue<std::pair<uint32_t, std::string>> _gain_dir_tracking;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    vector<float> default_gains;

    /// Buffer for accessing metadata
    Buffer* metadata_buf;

    /// Freq bin index, where the 0th is at 800MHz
    std::vector<freq_id_t> freq_idx;

    /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    /// Flag to control gains to be only loaded on request.
    std::atomic<bool> update_frb_flag;
    std::atomic<bool> update_tracking_flag;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of pulsar beams, should be 10
    uint32_t _num_beams;
    /// Number of frequencies per node
    uint32_t _num_local_freq;
    /// Frequency upchannelization factor
    uint32_t _upchan_factor;

    /// Fixed buffers used to hold gains separately from the kotekan buffers
    std::vector<float> frb_gains;
    std::vector<float> tracking_gains;

    /// implements `kotekan_gains_last_update_success`
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& gains_last_update_success_metric;

    /// implements `kotekan_gains_last_update_timestamp`
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>&
        gains_last_update_timestamp_metric;
};


#endif
