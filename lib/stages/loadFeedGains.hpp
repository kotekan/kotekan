
#ifndef LOAD_FEED_GAINS_HPP
#define LOAD_FEED_GAINS_HPP

#include "Config.hpp"            // for Config
#include "N2Util.hpp"            // for frameID
#include "Stage.hpp"             // for Stage
#include "Telescope.hpp"         // for freq_id_t
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily

#include "json.hpp" // for json

#include <mutex>    // for lock_guard, mutex
#include <queue>    // for queue
#include <stdint.h> // for int32_t, uint8_t, int16_t, uint32_t
#include <string>   // for string
#include <unordered_map>
#include <utility> // for pair
#include <vector>  // for vector

/**
 * @class loadFeedGains
 * @brief Read in new gain files for a beamformer when available.
 *
 * New gains are indicated by an updatable config endpoint. Files are loaded
 * per-frequency and merged into a single buffer, with frequency as the outer axis.
 *
 * In order to support beamformers with individual gains for different beams, this
 * stage can produce an arbitrary number of output buffers, each corresponding to
 * a single beam. This is intended to be used in conjunction with the `processFeedGains`
 * stage, which merges all the individual buffers.
 *
 * This stage emits a buffer containing gains for a given beam whenever an endpoint
 * is updated.
 *
 * @par Buffers
 * @buffer in_buf Arbitrary buffer used to collect metadata. Must contain _all_ local frequencies.
 *     @buffer_format any
 *     @buffer_shape any
 *     @buffer_metadata chordMetadata
 * @buffer gain_buf_<n> Each buffer is an array of arbitrary gains for a given beam
 *     @buffer_format float32
 *     @buffer_shape [freq][element][real/imag]
 *     @buffer_metadata chordMetadata
 *
 * @conf   num_elements         Int. Number of elements
 * @conf   num_beams            Int. Number of beams with unique gain files
 * @conf   num_local_freq       Int. Number of frequencies per node
 * @conf   default_gains        Float array (default 0 + 0j). Default gain value if gain file is
 *                                missing.
 * @conf   filename_template    String: Template for a gain file name. Must contain a formatter that
 *                                can handle a 4-digit integer, corresponding to a frequency ID.
 *
 * @par Metrics
 * @metric kotekan_gains_last_update_success
 *     Flag with value 1 if the last gains update succeeded, 0 if failed.
 *     Labels: `type` (string)
 *             `beam_id` (uint32_t)
 * @metric kotekan_gains_last_update_timestamp
 *     Timestamp of last gains update.
 *     Labels: `type` (string)
 *             `beam_id` (uint32_t)
 *
 * The gain path is registered as a subscriber to an updatable config block.
 * Updates are provided on per-beam endpoints, each with one directory
 * path for the corresponding beam: `{"gain_directory": "path_for_beam"}`.
 *
 * @author Liam Gray
 *
 */
class loadFeedGains : public kotekan::Stage {
public:
    /// Constructor
    loadFeedGains(kotekan::Config& config_, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~loadFeedGains();

    void main_thread() override;

    /// Endpoint for providing new directory path for gain updates
    bool update_gains_callback(nlohmann::json& json, const uint32_t beam_id);

    /// Load new gains from the provided directory
    bool update_gains(Buffer* buf, N2::frameID frame_id, std::pair<uint32_t, std::string> beam);

private:
    /// Lock when changing status of gain callback queue
    std::mutex mtx;

    /// Gain buffer
    std::unordered_map<uint32_t, Buffer*> gain_buffers;
    /// Metadata buffer, accessed once at startup
    Buffer* metadata_buf;

    /// Gain file name. Must contain a format identifier for
    /// a 4+ digit integer. Historically, this was "quick_gains_%04d_reordered.bin"
    std::string filename_template;

    /// Queue containing (beam_id, file_directory) pairs
    std::queue<std::pair<uint32_t, std::string>> gain_directories;

    /// Frequency IDs
    std::vector<freq_id_t> freq_id;

    /// Gain parameters
    std::vector<float> default_gains;

    /// Record the name of this beamformer type
    std::string beamformer_type;

    /// Buffer shape parameters
    uint32_t num_elements;
    uint32_t num_beams;
    uint32_t num_local_freq;

    /// Prometheus metrics
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& last_update_success_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& last_update_timestamp_metric;
};


#endif
