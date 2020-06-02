#ifndef APPLY_GAINS_HPP
#define APPLY_GAINS_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "SynchronizedQueue.hpp" // for SynchronizedQueue
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, state_id_t
#include "prometheusMetrics.hpp" // for Counter, Gauge
#include "restClient.hpp"        // for restClient
#include "updateQueue.hpp"       // for updateQueue
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for cfloat, frameID

#include "json.hpp" // for json

#include <atomic>       // for atomic
#include <ctime>        // for timespec, size_t
#include <map>          // for map
#include <mutex>        // for mutex
#include <optional>     // for optional
#include <shared_mutex> // for shared_mutex
#include <stdint.h>     // for uint32_t, uint64_t
#include <string>       // for string
#include <tuple>        // for tuple
#include <utility>      // for pair
#include <vector>       // for vector

/**
 * @class applyGains
 * @brief Receives gains and apply them to the output buffer.
 *
 * This stage registers as a subscriber to an updatable config block. The
 * full name of the block should be defined in the value @c updatable_block
 *
 * Gain updates *must* match the frequencies expected to be present in the
 * input stream. That is there must be exactly as many frequencies in the gain
 * update as there are in the `freqState` attached to the input stream.
 * The number of elements must also match those on the incoming
 * stream.
 *
 * The number of frequencies and inputs is locked in
 *
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format VisFrameView.
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The output stream.
 *         @buffer_format VisFrameView.
 *         @buffer_metadata VisMetadata
 *
 * @conf   num_elements     Int.    The number of elements (i.e. inputs) in the
 *                                  correlator data.
 * @conf   updatable_block  String. The full name of the updatable_block that
 *                                  will provide new flagging values (e.g. "/dynamic_block/gains").
 * @conf   gains_dir        String. The path to the directory holding the gains file.
 * @conf   broker_host      String. Calibration broker host.
 * @conf   broker_port      Int.    Calibration broker port.
 * @conf   read_from_file   Bool, default false. Whether to read the gains from file or
 *                                  fetch them over the network.
 * @conf   tcombine         Double. Time (in seconds) over which to combine old and new gains to
 *                                  prevent discontinuities. Default is 5 minutes.
 * @conf   num_kept_updates Int.    The number of gain updates stored in a FIFO.
 * @conf   num_threads      Int.    Number of threads to run. Default is 1.
 *
 * @par Metrics
 * @metric kotekan_applygains_late_update_count The number of updates received
 *     too late (The start time of the update is older than the currently
 *     processed frame).
 * @metric kotekan_applygains_late_frame_count The number of frames received
 *     late (The frames timestamp is older then all start times of stored
 *     updates).
 * @metric kotekan_applygains_update_age_seconds The time difference in
 *     seconds between the current frame being processed and the time stamp of
 *     the gains update being applied.
 *
 * @author Mateus Fandino, Tristan Pinsonneault-Marotte and Richard Shaw
 */
class applyGains : public kotekan::Stage {

public:
    /// Default constructor
    applyGains(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Main loop for the stage
    void main_thread() override;

    /// Callback function to receive updates on timestamps from configUpdater
    bool receive_update(nlohmann::json& json);

private:
    // An internal type for holding the actual gains
    struct GainData {
        std::vector<std::vector<cfloat>> gain;
        std::vector<std::vector<float>> weight;
    };

    // An internal type for holding all information about the gain update
    struct GainUpdate {
        GainData data;
        double transition_interval;
        state_id_t state_id;
    };

    // Parameters saved from the config files

    /// Path to gains directory
    std::string gains_dir;

    /// Number of gains updates to maintain
    uint64_t num_kept_updates;

    /// Host and port of calibration broker
    std::string broker_host;
    unsigned int broker_port;

    /// Whether to read gains from file or over network
    bool read_from_file;

    /// The gains and when to start applying them in a FIFO (len set by config)
    updateQueue<GainUpdate> gains_fifo;

    /// Input buffer to read from
    Buffer* in_buf;
    /// Output buffer with gains applied
    Buffer* out_buf;

    /// Mutex to protect access to gains and freq map
    std::shared_mutex gain_mtx;
    std::shared_mutex freqmap_mtx;

    /// Timestamp of the current frame
    std::atomic<timespec> ts_frame{{0, 0}};

    /// Entrancepoint for n threads. Each thread takes frames with a
    /// different frame_id from the buffer and applies gains.
    void apply_thread();

    /// Thread for getting gains from cal broker
    void fetch_thread();

    /// Number of parallel threads accessing the same buffers (default 1)
    uint32_t num_threads;

    /// Input frame ID, shared by apply threads.
    frameID frame_id_in;

    /// Output frame ID, shared by apply threads.
    frameID frame_id_out;

    /// Mutex protecting shared frame IDs.
    std::mutex m_frame_ids;

    // Prometheus metrics
    kotekan::prometheus::Gauge& update_age_metric;
    kotekan::prometheus::Counter& late_update_counter;
    kotekan::prometheus::Counter& late_frames_counter;

    /// Read the gain file from disk
    std::optional<GainData> read_gain_file(std::string update_id) const;

    /// Fetch gains from calibration broker
    std::optional<GainData> fetch_gains(std::string tag) const;

    /// Used to indicate to other threads when incoming data has arrived and we
    /// are processing
    std::atomic<bool> started = false;

    /// Queue used to send updates into the fetch thread
    using update_t = std::tuple<std::string, double, double, bool>;
    SynchronizedQueue<update_t> update_fetch_queue;


    /// REST client for communication with cal broker
    restClient& client;


    /// Wait until the first frame comes in and read its metadata to determine
    /// what frequencies and inputs we will get
    void initialise();

    /// Calculate the gain for this time and frequency
    /// Returns false if there was no appropriate gain and we need to skip
    std::tuple<bool, double, state_id_t> calculate_gain(double timestamp, uint32_t freq_id,
                                                        std::vector<cfloat>& gain,
                                                        std::vector<cfloat>& gain_conj,
                                                        std::vector<float>& weight_factor) const;

    /// Test that the frame is valid. On failure it will call FATAL_ERROR and
    /// return false
    bool validate_frame(const VisFrameView& frame) const;

    /// Test that the gain is valid. On failure it will call FATAL_ERROR and
    /// return false. Gains failing this *should* have already been rejected,
    /// except the initial gains which can't be checked.
    bool validate_gain(const GainData& frame) const;


    // Check if file to read exists
    bool fexists(const std::string& filename) const;

    // Save the number of frequencies and elements for checking gain updates
    // against
    std::optional<size_t> num_freq;
    std::optional<size_t> num_elements;
    std::optional<size_t> num_prod;

    // Mapping from frequency ID to index (in the gain file)
    std::map<uint32_t, uint32_t> freq_map;

    // Map from the state being applied and input dataset to the output dataset.
    // This is used to keep track of the labels we should be applying for
    // timesamples coming out of order.
    std::map<std::pair<state_id_t, dset_id_t>, dset_id_t> output_dataset_ids;
};


#endif
