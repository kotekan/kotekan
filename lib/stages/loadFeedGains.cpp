#include "loadFeedGains.hpp"

#include "Config.hpp" // for Config
#include "N2Util.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"    // for Telescope, FREQ_ID_NOT_SET
#include "buffer.hpp"       // for Buffer
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for WARN, INFO, DEBUG
#include "visUtil.hpp"        // for current_time

#include <chrono>
#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <regex>
#include <stdio.h> // for fclose, fopen, fread, snprintf, FILE
#include <string>
#include <sys/types.h> // for uint
#include <thread>      // for sleep_for


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(loadFeedGains);

loadFeedGains::loadFeedGains(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&loadFeedGains::main_thread, this)),
    last_update_success_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_success", unique_name, {"type", "beam_id"})),
    last_update_timestamp_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_gains_last_update_timestamp", unique_name, {"type", "beam_id"})) {
    // Apply config
    beamformer_type = config.get<std::string>(unique_name, "beamformer_type");
    num_elements = config.get<uint32_t>(unique_name, "num_elements");
    num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    num_beams = config.get_default<uint32_t>(unique_name, "num_beams", 1);

    // gain file name
    filename_template = config.get<std::string>(unique_name, "filename_template");
    // validate that the template can accomodate a decimal frequency index
    std::regex int_format(R"(%[-+ #0]*\d*[diu])");
    if (!std::regex_search(filename_template, int_format)) {
        FATAL_ERROR("invalid filename template: {:s}", filename_template);
    }

    // default gains have shape {Re, Im}
    default_gains =
        config.get_default<std::vector<float>>(unique_name, "default_gains", {0.0, 0.0});

    metadata_buf = get_buffer("in_buf");
    metadata_buf->register_consumer(unique_name);
    // Initialize the vector holding freq ID
    freq_id.resize(num_local_freq, FREQ_ID_NOT_SET);

    // Register as a producer for all gain buffers and check that their
    // dimensionalitites match
    json in_buf_list = config.get_value(unique_name, "gain_buffers");
    if (in_buf_list.size() != num_beams) {
        FATAL_ERROR("Expected {:d} gain buffers to match num_beams, got {:d}", num_beams,
                    in_buf_list.size());
    }

    for (size_t beam_id = 0; beam_id < num_beams; beam_id++) {
        const std::string buf_name = in_buf_list.at(beam_id).get<std::string>();
        Buffer* buf = buffer_container.get_buffer(buf_name);

        gain_buffers.emplace(beam_id, buf);
        buf->register_producer(unique_name);

        if (beam_id > 0 && buf->frame_size != gain_buffers.at(0)->frame_size) {
            FATAL_ERROR("Input buffers have different frame sizes: {:d} != {:d}", buf->frame_size,
                        gain_buffers.at(0)->frame_size);
        }

        if (buf->frame_size != num_local_freq * num_elements * 2 * sizeof(float)) {
            FATAL_ERROR("Input buffer does not have the expected size. Expected {:d}, got {:d}",
                        num_local_freq * num_elements * 2 * sizeof(float), buf->frame_size);
        }
        // Set frame description
        buf->ensure_frame_desc(kotekan::NDArray<kotekan::GetType_t<kotekan::float32>, 3>::describe(
            "gain",
            {static_cast<ptrdiff_t>(num_local_freq), static_cast<ptrdiff_t>(num_elements),
             static_cast<ptrdiff_t>(2)},
            {"F", "E", "C"}, {1, 1, 1}));
    }

    // Set up callbacks for each beam_id
    using namespace std::placeholders;
    std::string endpoint_name = "updatable_config/" + beamformer_type + "_gain";
    for (uint32_t beam_id = 0; beam_id < num_beams; beam_id++) {
        configUpdater::instance().subscribe(
            config.get<std::string>(unique_name, endpoint_name.c_str()) + "/"
                + std::to_string(beam_id),
            [beam_id, this](nlohmann::json& json_msg) -> bool {
                return update_gains_callback(json_msg, beam_id);
            });
    }
}

loadFeedGains::~loadFeedGains() {}

bool loadFeedGains::update_gains_callback(nlohmann::json& json, const uint32_t beam_id) {
    std::string dir;

    try {
        dir = json.at("gain_directory").get<std::string>();
        INFO("[{:s}] Updating gains from {:s} for beam {:d}", beamformer_type, dir, beam_id);
    } catch (std::exception const& e) {
        WARN("[{:s}] Failed to parse gain_directory {:s} for beam {:d}", beamformer_type, e.what(),
             beam_id);
        return false;
    }

    // Record the beam-specific update
    {
        std::lock_guard<std::mutex> lock(mtx);
        gain_directories.push(std::make_pair(beam_id, dir));
    }

    return true;
}

bool loadFeedGains::update_gains(Buffer* buf, N2::frameID frame_id,
                                 std::pair<uint32_t, std::string> beam) {
    // wait for an empty frame on this buffer
    float* frame = (float*)buf->wait_for_empty_frame(unique_name, frame_id);
    if (frame == nullptr) {
        return false;
    }

    double start_time = current_time();
    FILE* ptr_myfile;

    size_t fstride = num_elements * 2;

    bool any_failed = false;

    uint32_t beam_id = beam.first;

    for (size_t i = 0; i < freq_id.size(); ++i) {
        auto fid = freq_id[i];
        bool ffailed = false;
        auto foffset = i * fstride;

        // Construct the gain file name for this freq
        std::string ftemplate = "%s/" + filename_template;
        auto required_chars = snprintf(nullptr, 0, ftemplate.c_str(), beam.second.c_str(), fid);
        std::string filename;

        if (required_chars < 0) {
            WARN("[{:s}] Failed to format gain file at {:s} for frequency id {:d}", beamformer_type,
                 beam.second.c_str(), fid);
            ptr_myfile = nullptr;
        } else {
            std::vector<char> fname_buf(static_cast<size_t>(required_chars) + 1);
            int namelen = snprintf(fname_buf.data(), fname_buf.size(), ftemplate.c_str(),
                                   beam.second.c_str(), fid);

            if (namelen < 0 || namelen >= required_chars + 1) {
                WARN("[{:s}] Failed to format gain file at {:s} for frequency id {:d} - error "
                     "writing filename buffer (expected {:d} characters, wrote {:d})",
                     beamformer_type, beam.second.c_str(), fid, required_chars + 1, namelen);
                ptr_myfile = nullptr;
            } else {
                filename.assign(fname_buf.data());
                ptr_myfile = fopen(filename.c_str(), "rb");

                if (ptr_myfile == nullptr) {
                    WARN("[{:s}] Failed to open gain file {:s}", beamformer_type, filename);
                } else {
                    INFO("[{:s}] Loaded gains from {:s}", beamformer_type, filename);
                }
            }
        }

        if (ptr_myfile == nullptr) {
            any_failed = true;
            ffailed = true;
        } else {
            auto bytes_read = fread(frame + foffset, sizeof(float), 2 * num_elements, ptr_myfile);
            fclose(ptr_myfile);

            if (bytes_read != 2 * num_elements) {
                WARN("[{:s}] Gain file ({:s}) wasn't long enough! Using default gains",
                     beamformer_type, filename);
                any_failed = true;
                ffailed = true;
            };
        }

        // If the read failed at any point, just use default gains, but still
        // return true
        if (ffailed) {
            float* dst = frame + foffset;
            for (uint j = 0; j < 2 * num_elements; j += 2) {
                dst[j] = default_gains[0];
                dst[j + 1] = default_gains[1];
            }
        }
    }

    // Update metrics once after all freqs pass
    last_update_timestamp_metric.labels({beamformer_type, std::to_string(beam_id)}).set(start_time);

    if (any_failed) {
        last_update_success_metric.labels({beamformer_type, std::to_string(beam_id)}).set(0);
        WARN("[{:s}] Failed to load some gains for beam {:d}", beamformer_type, beam_id);
    } else {
        last_update_success_metric.labels({beamformer_type, std::to_string(beam_id)}).set(1);
        INFO("[{:s}] Successfully updated gains for beam {:d}", beamformer_type, beam_id);
    }

    return true;
}

void loadFeedGains::main_thread() {
    // Create frame IDs for output buffers
    std::unordered_map<uint32_t, N2::frameID> gain_buffer_frame_ids;
    for (uint32_t beam_id = 0; beam_id < gain_buffers.size(); ++beam_id) {
        // new default frameID is created for each buffer
        gain_buffer_frame_ids.emplace(beam_id, gain_buffers.at(beam_id));
    }

    // grab frequency information from the metadata buffer.
    // don't need to track metadata buffer ID here - they just get used
    // once at startup
    uint8_t* meta_frame = metadata_buf->wait_for_full_frame(unique_name, 0);
    if (meta_frame == nullptr)
        return;

    auto meta_freq_idx = get_chord_metadata(metadata_buf, 0)->get_coarse_freq();
    if (freq_id.size() != meta_freq_idx.size()) {
        FATAL_ERROR("Number of frequencies received does not match expectation: {:d} != {:d}",
                    meta_freq_idx.size(), freq_id.size());
    };

    // Assign the frequency indices and MHz values
    for (size_t f = 0; f < freq_id.size(); ++f) {
        freq_id_t fid = meta_freq_idx[f];
        freq_id[f] = fid;
    }

    metadata_buf->mark_frame_empty(unique_name, 0);
    metadata_buf->unregister_consumer(unique_name);

    while (!stop_thread) {
        // wait until there's something to update
        std::pair<uint32_t, std::string> beam;
        {
            std::unique_lock<std::mutex> lock(mtx);
            if (gain_directories.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            } else {
                beam = gain_directories.front();
                gain_directories.pop();
            }
        }

        // get the buffer, add metadata, fill with gains, and emit a buffer
        Buffer* buf = gain_buffers.at(beam.first);
        N2::frameID& frame_id = gain_buffer_frame_ids.at(beam.first);

        // Set metadata from the frame desc, and other metadata
        buf->allocate_new_metadata_object(frame_id);
        auto meta = get_chord_metadata(buf, frame_id);
        meta->set_from_frame_desc(buf->get_frame_desc<kotekan::GenericNDArray>());
        meta->set_name("gain");
        meta->set_coarse_freq(meta_freq_idx);
        // Verify that frame desc and metadata match
        meta->check_frame_desc(buf->get_frame_desc<kotekan::GenericNDArray>());

        // load gains into the buffer, returning in the received frame
        // is invalid
        if (!update_gains(buf, frame_id, beam)) {
            return;
        }

        buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
    }
}
