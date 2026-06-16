#include "processGains.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"      // for Telescope, FREQ_ID_NOT_SET
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for WARN, INFO, DEBUG
#include "visUtil.hpp"        // for current_time

#include <exception>   // for exception
#include <functional>  // for bind, function, _1
#include <memory>      // for __shared_ptr_access, shared_ptr
#include <stdio.h>     // for fclose, fopen, fread, snprintf, FILE
#include <sys/types.h> // for uint


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(processGains);

// clang-format off

// Request gain file re-parse with e.g.
// FRB
// curl localhost:12048/frb_gain -X POST -H 'Content-Type: application/json' -d '{"gain_dir":"the_new_path"}'
// Tracking Beamformer
// curl localhost:12048/updatable_config/tracking_gain -X POST -H 'Content-Type: application/json' -d
// '{"gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'
//
// clang-format on

processGains::processGains(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&processGains::main_thread, this)),
    gain_frb_frame_id(get_buffer("gain_frb_buf")),
    gain_tracking_frame_id(get_buffer("gain_tracking_buf")),
    gains_last_update_success_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_success", unique_name, {"type", "beam_id"})),
    gains_last_update_timestamp_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_timestamp", unique_name, {"type", "beam_id"})) {
    // Apply config.
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_beams = config.get<uint32_t>(unique_name, "num_beams");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _upchan_factor = config.get_default<uint32_t>(unique_name, "frb_upchan_factor", 1);
    scaling = config.get_default<float>(unique_name, "frb_scaling", 1.0);

    vector<float> dg = {0.0, 0.0}; // re,im
    default_gains = config.get_default<std::vector<float>>(unique_name, "frb_missing_gains", dg);

    metadata_buf = get_buffer("in_buf");
    metadata_buf->register_consumer(unique_name);
    // Initialize the vector holding freq ID
    freq_idx.resize(_num_local_freq, FREQ_ID_NOT_SET);

    // Gain for FRB
    gain_frb_buf = get_buffer("gain_frb_buf");
    gain_frb_buf->register_producer(unique_name);
    update_frb_flag.store(false);

    // Gain for Tracking Beamformer
    gain_tracking_buf = get_buffer("gain_tracking_buf");
    gain_tracking_buf->register_producer(unique_name);
    update_tracking_flag.store(false);

    // Check that buffers are as-expected
    auto frb_size = _num_elements * _num_local_freq * _upchan_factor * 2 * sizeof(float);
    if (gain_frb_buf->frame_size != frb_size) {
        FATAL_ERROR("gain_frb_buf does not have the expected size. Expected {:d}, got {:d}",
                    frb_size, gain_frb_buf->frame_size);
    }

    auto tracking_size = _num_elements * _num_local_freq * _num_beams * 2 * sizeof(float);
    if (gain_tracking_buf->frame_size != tracking_size) {
        FATAL_ERROR("gain_tracking_buf does not have the expected size. Expected {:d}, got {:d}",
                    tracking_size, gain_tracking_buf->frame_size);
    }

    // Set frame descriptions for each produced
    gain_frb_buf->allocate_ndarray_frame_desc<kotekan::GetType_t<kotekan::float32>, 4>(
        "gain_frb",
        {static_cast<ptrdiff_t>(_num_local_freq), static_cast<ptrdiff_t>(_upchan_factor),
         static_cast<ptrdiff_t>(_num_elements), (ptrdiff_t)2},
        {"F", "U", "E", "ReIm"});

    gain_tracking_buf->allocate_ndarray_frame_desc<kotekan::GetType_t<kotekan::float32>, 4>(
        "gain_tracking",
        {static_cast<ptrdiff_t>(_num_local_freq), static_cast<ptrdiff_t>(_num_beams),
         static_cast<ptrdiff_t>(_num_elements), (ptrdiff_t)2},
        {"F", "R", "E", "ReIm"});

    // Allocate permanent buffers to hold the loaded gains in memory. These will only
    // be updated whenever the gains are updated, and get repeatedly copied
    // into received kotekan buffers
    frb_gains = std::vector<float>(gain_frb_buf->frame_size / sizeof(float), 0.0f);
    tracking_gains = std::vector<float>(gain_tracking_buf->frame_size / sizeof(float), 0.0f);

    using namespace std::placeholders;
    // listen for gain updates FRB
    std::string gainfrb =
        config.get_default<std::string>(unique_name, "updatable_config/gain_frb", "");
    if (gainfrb.length() > 0)
        configUpdater::instance().subscribe(
            gainfrb, std::bind(&processGains::update_gains_frb_callback, this, _1));

    // Listen for gain updates Tracking Beamformer
    using namespace std::placeholders;
    for (uint32_t beam_id = 0; beam_id < _num_beams; beam_id++) {
        configUpdater::instance().subscribe(
            config.get<std::string>(unique_name, "updatable_config/gain_tracking") + "/"
                + std::to_string(beam_id),
            [beam_id, this](nlohmann::json& json_msg) -> bool {
                return update_gains_tracking_callback(json_msg, beam_id);
            });
    }
}

processGains::~processGains() {}

bool processGains::update_gains_frb_callback(nlohmann::json& json) {
    std::string dir;
    try {
        dir = json.at("gain_dir").get<std::string>();
    } catch (std::exception& e) {
        WARN("[FRB] Failed to parse gain_dir {:s}", e.what());
        return false;
    }

    // Lock and update the gain directory
    {
        std::lock_guard<std::mutex> lock(mux);
        _gain_dir_frb = dir;
    }
    // record that we need to load and update gain buffer
    update_frb_flag.store(true);

    return true;
}

bool processGains::update_gains_tracking_callback(nlohmann::json& json, const uint32_t beam_id) {
    std::string dir;
    try {
        dir = json.at("gain_dir").get<std::string>();
        INFO("[Tracking Beamformer] Updating gains from {:s} for beam {:d}", dir, beam_id);
    } catch (std::exception const& e) {
        WARN("[Tracking Beamformer] Failed to parse gain_dir {:s} for beam {:d}", e.what(),
             beam_id);
        return false;
    }

    // Record the beam-specific update
    {
        std::lock_guard<std::mutex> lock(mux);
        _gain_dir_tracking.push(std::make_pair(beam_id, dir));
    }
    update_tracking_flag.store(true);

    return true;
}

void processGains::upchannelize_gain_frb() {
    if (_upchan_factor < 2) {
        return;
    }
    // duplicate the gains at each upchan = 0 bin for each freq
    size_t chan_stride = _num_elements * 2;
    size_t fstride = _upchan_factor * chan_stride;

    float* data_ptr = frb_gains.data();

    for (size_t f = 0; f < freq_idx.size(); ++f) {
        // pointer to the first fine channel of each coarse freq
        float* f_ptr = data_ptr + f * fstride;

        for (size_t u = 1; u < _upchan_factor; ++u) {
            // copy all elements from chan 0 into each subsequent
            // channel for this frequency
            float* u_ptr = f_ptr + u * chan_stride;
            // NB: implement any additional interpolation logic here
            std::copy_n(f_ptr, chan_stride, u_ptr);
        }
    }
}

void processGains::update_gain_frb() {
    if (!update_frb_flag.load()) {
        return;
    }

    // Lock and record the current gain directory
    std::string gain_directory;

    {
        std::lock_guard<std::mutex> lock(mux);
        gain_directory = _gain_dir_frb;
    }

    double start_time = current_time();
    FILE* ptr_myfile;

    size_t fstride = _upchan_factor * _num_elements * 2; // real and complex stored separately
    bool any_failed = false;                             // set if any freq can't be read

    // Each frequency is provided in a different file
    for (size_t i = 0; i < freq_idx.size(); ++i) {
        auto fid = freq_idx[i];
        bool ffailed = false; // set if this frequency can't be read
        auto foffset = i * fstride;

        // Construct the gain file name for this freq
        auto required_chars =
            snprintf(nullptr, 0, "%s/quick_gains_%04d_reordered.bin", gain_directory.c_str(), fid)
            + 1; // +1 for terminating null char
        std::vector<char> fname_buf(required_chars);
        char* filename = fname_buf.data();
        auto namelen = snprintf(filename, required_chars, "%s/quick_gains_%04d_reordered.bin",
                                gain_directory.c_str(), fid);

        if (namelen != required_chars) {
            WARN("Failed to format gain filename for frequency {:d}", fid);
            ptr_myfile = nullptr;
        } else {
            INFO("[FRB] Loading gains from {:s}", filename);
            ptr_myfile = fopen(filename, "rb");
        }

        if (ptr_myfile == nullptr) {
            WARN("Failed to open gain file {:s}", filename);
            any_failed = true;
            ffailed = true;
        } else {
            // Each file only contains gains for a single coarse frequency channel, so read into
            // the first upchannelization bin for each frequency. Upchannelization happens
            // once all frequencies have been loaded and inserted into the permanent buffer.
            auto bytes_read =
                fread(frb_gains.data() + foffset, sizeof(float), 2 * _num_elements, ptr_myfile);

            fclose(ptr_myfile);
            if (bytes_read != 2 * _num_elements) {
                WARN("Gain file ({:s}) wasn't long enough! Using default gains", filename);
                any_failed = true;
                ffailed = true;
            }
        }

        // If there was a failure to read the data, use the default gains
        if (ffailed) {
            for (uint j = 0; j < 2 * _num_elements; j += 2) {
                frb_gains[foffset + j] = default_gains[0];
                frb_gains[foffset + j + 1] = default_gains[1];
            }
        }
        // Apply scaling factor, constant for real and imag
        for (uint j = 0; j < 2 * _num_elements; j++) {
            frb_gains[foffset + j] *= scaling;
        }
    }

    // Upchannelize the FRB gains in-place
    upchannelize_gain_frb();

    // Just records if _any_ freq failed.
    // NB: should we include freq labels here?
    if (any_failed) {
        gains_last_update_success_metric.labels({"frb", ""}).set(0);
        WARN("Failed to update FRB gains for some frequencies");
    } else {
        gains_last_update_success_metric.labels({"frb", ""}).set(1);
        INFO("Successfully updated FRB gains");
    }

    gains_last_update_timestamp_metric.labels({"frb", ""}).set(start_time);

    update_frb_flag.store(false);
}

/// Reads the tracking gains for each beam, with a different configurable
/// endpoint for each beam.
void processGains::update_gain_tracking() {
    if (!update_tracking_flag.load()) {
        return;
    }

    // If we should update, wait for a moment in case multiple beam updates
    // are received in a short period of time (so that we can try to handle
    // them in one pass)
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Lock and record the current directories
    queue<std::pair<uint32_t, std::string>> directories;
    {
        std::lock_guard<std::mutex> lock(mux);
        directories = _gain_dir_tracking;
    }

    // drain the queue into an unordered map, dealing with the case
    // that there have been multiple updates for the same beam since the
    // last pass through the queue.
    std::unordered_map<uint32_t, std::string> updates;

    while (directories.size() > 0) {
        std::pair<uint32_t, std::string> entry = directories.front();
        // remove item from queue
        directories.pop();
        // insert and overwrite if the update already exists
        updates[entry.first] = entry.second;
    }

    double start_time = current_time();
    FILE* ptr_myfile;

    size_t fstride = _num_elements * _num_beams * 2;

    for (const auto& beam : updates) {
        bool any_failed = false;

        uint32_t beam_id = beam.first;
        std::string beam_label = "tracking_beam_" + std::to_string(beam_id);

        for (size_t i = 0; i < freq_idx.size(); ++i) {
            auto fid = freq_idx[i];
            bool ffailed = false;
            auto foffset = i * fstride;

            // Construct the gain file name for this freq
            auto required_chars =
                snprintf(nullptr, 0, "%s/quick_gains_%04d_reordered.bin", beam.second.c_str(), fid)
                + 1;
            std::vector<char> fname_buf(required_chars);
            char* filename = fname_buf.data();
            auto namelen = snprintf(filename, required_chars, "%s/quick_gains_%04d_reordered.bin",
                                    beam.second.c_str(), fid);

            if (namelen < required_chars) {
                WARN("Failed to format gain filename for frequency {:d}", fid);
                ptr_myfile = nullptr;
            } else {
                INFO("[Tracking Beamformer] Loading gains from {:s}", filename);
                ptr_myfile = fopen(filename, "rb");
            }

            if (ptr_myfile == nullptr) {
                WARN("Failed to open gain file {:s}", filename);
                any_failed = true;
                ffailed = true;
            } else {
                auto bytes_read =
                    fread(tracking_gains.data() + foffset + beam_id * _num_elements * 2,
                          sizeof(float), 2 * _num_elements, ptr_myfile);
                fclose(ptr_myfile);

                if (bytes_read != 2 * _num_elements) {
                    WARN("Gain file ({:s}) wasn't long enough! Using default gains", filename);
                    any_failed = true;
                    ffailed = true;
                };
            }

            // If the read failed at any point, just use default gains
            if (ffailed) {
                for (uint j = 0; j < _num_elements; j++) {
                    size_t idx = foffset + (beam_id * _num_elements + j) * 2;
                    tracking_gains[idx] = default_gains[0];
                    tracking_gains[idx + 1] = default_gains[1];
                }
            }
        }
        // Update metrics once after all freqs pass
        gains_last_update_timestamp_metric.labels({"tracking", std::to_string(beam_id)})
            .set(start_time);

        if (any_failed) {
            gains_last_update_success_metric.labels({"tracking", std::to_string(beam_id)}).set(0);
            WARN("Failed to load some tracking beam gains");
        } else {
            gains_last_update_success_metric.labels({"tracking", std::to_string(beam_id)}).set(1);
            INFO("Successfully updated tracking beam gains");
        }
    }
    update_tracking_flag.store(false);
}

void processGains::main_thread() {
    // don't need to track metadata buffer ID here - they just get used
    // once at startup
    uint8_t* frame = metadata_buf->wait_for_full_frame(unique_name, 0);
    if (frame == nullptr)
        return;

    auto meta_freq_idx = get_chord_metadata(metadata_buf, 0)->get_coarse_freq();
    if (freq_idx.size() != meta_freq_idx.size()) {
        FATAL_ERROR("Number of frequencies received does not match expectation: {:d} != {:d}",
                    meta_freq_idx.size(), freq_idx.size());
    };

    // Assign the frequency indices and MHz values
    for (size_t f = 0; f < freq_idx.size(); ++f) {
        freq_id_t fid = meta_freq_idx[f];
        freq_idx[f] = fid;
    }

    metadata_buf->mark_frame_empty(unique_name, 0);
    metadata_buf->unregister_consumer(unique_name);

    while (!stop_thread) {
        // Check for gain updates. No-op if nothing has changed
        update_gain_frb();
        update_gain_tracking();

        // Wait for an empty frb gain buffer
        float* out_frame_frb =
            (float*)gain_frb_buf->wait_for_empty_frame(unique_name, gain_frb_frame_id);
        if (out_frame_frb == nullptr) {
            return;
        }
        // Set metadata from the frame desc, and other metadata
        gain_frb_buf->allocate_new_metadata_object(gain_frb_frame_id);
        auto frb_meta = get_chord_metadata(gain_frb_buf, gain_frb_frame_id);
        frb_meta->set_from_frame_desc(gain_frb_buf->get_ndarray_frame_desc());
        frb_meta->set_name("gain_frb");
        // Verify that frame desc and metadata match
        frb_meta->check_frame_desc(gain_frb_buf->get_ndarray_frame_desc());
        {
            // Make sure gains don't change while copying
            std::lock_guard<std::mutex> lock(mux);
            std::copy(frb_gains.begin(), frb_gains.end(), out_frame_frb);
        }
        gain_frb_buf->mark_frame_full(unique_name, gain_frb_frame_id);
        gain_frb_frame_id++;

        // Repeat with tracking buffer
        float* out_frame_tracking =
            (float*)gain_tracking_buf->wait_for_empty_frame(unique_name, gain_tracking_frame_id);
        if (out_frame_tracking == nullptr) {
            return;
        }
        // Set metadata from the frame desc, and other metadata
        gain_tracking_buf->allocate_new_metadata_object(gain_tracking_frame_id);
        auto tracking_meta = get_chord_metadata(gain_tracking_buf, gain_tracking_frame_id);
        tracking_meta->set_from_frame_desc(gain_tracking_buf->get_ndarray_frame_desc());
        tracking_meta->set_name("gain_tracking");
        // Verify that frame desc and metadata match
        tracking_meta->check_frame_desc(gain_tracking_buf->get_ndarray_frame_desc());
        {
            // Copy to output buf, holding the lock to make sure the stored
            // gains do not change
            std::lock_guard<std::mutex> lock(mux);
            std::copy(tracking_gains.begin(), tracking_gains.end(), out_frame_tracking);
        }
        gain_tracking_buf->mark_frame_full(unique_name, gain_tracking_frame_id);
        gain_tracking_frame_id++;
    }
}
