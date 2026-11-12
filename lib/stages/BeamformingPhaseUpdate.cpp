#include "BeamformingPhaseUpdate.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"       // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "chordMetadata.hpp"
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG
#include "visUtil.hpp"        // for frameID
#include "util.h"             // for hex_dump


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

STAGE_CONSTRUCTOR(BeamformingPhaseUpdate) {

    // Register on buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    if (config.exists(unique_name.c_str(), "gains_buf")) {
        gains_buf = get_buffer("gains_buf");
        gains_buf->register_consumer(unique_name);
    }

    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_beams = config.get<uint32_t>(unique_name, "num_beams");
    _beam_offset = config.get_default<uint32_t>(unique_name, "beam_offset", 0);
    _num_local_beams = config.get_default<uint32_t>(unique_name, "num_local_beams", _num_beams);

    if (_beam_offset + _num_local_beams > _num_beams) {
        throw std::runtime_error("The offset + num_local_beams is larger than the total number of beams");
    }

    _inst_lat = config.get<double>(unique_name, "inst_lat");
    _inst_long = config.get<double>(unique_name, "inst_long");

    _num_local_freq = config.get<double>(unique_name, "num_local_freq");

    frequencies_in_frame.resize(_num_local_freq);

    // Get feed locations.
    feed_locations =
        config.get<std::vector<std::pair<double, double>>>(unique_name, "feed_positions");

    // Listen for DUT1 (UT1 - UTC) offset.
    std::string DUT1 =
        config.get_default<std::string>(unique_name, "updatable_config/DUT1", "");
    if (DUT1.length() > 0)
        kotekan::configUpdater::instance().subscribe(
            DUT1, std::bind(&BeamformingPhaseUpdate::update_DUT1, this, std::placeholders::_1));

    // Register function to listen for new beam, and update ra and dec.
    using namespace std::placeholders;
    for (uint32_t beam_id = 0; beam_id < _num_beams; beam_id++) {
        kotekan::configUpdater::instance().subscribe(
            config.get<std::string>(unique_name, "updatable_config/tracking_pt") + "/"
                + std::to_string(beam_id),
            [beam_id, this](nlohmann::json& json_msg) -> bool {
                return tracking_update_callback(json_msg, beam_id);
            });
    }
}

bool BeamformingPhaseUpdate::tracking_update_callback(nlohmann::json& json, const uint8_t beam_id) {
    {
        std::lock_guard<std::mutex> lock(beam_lock);
        try {
            _beam_coord.right_ascension[beam_id] = json.at("ra").get<float>();
        } catch (std::exception const& e) {
            WARN("[TRACKING] Pointing update failed to read RA from beam: {:d}. {:s}", beam_id,
                 e.what());
            return false;
        }
        try {
            _beam_coord.declination[beam_id] = json.at("dec").get<float>();
        } catch (std::exception const& e) {
            WARN("[TRACKING] Pointing update failed to read DEC from beam: {:d}. {:s}", beam_id,
                 e.what());
            return false;
        }
        try {
            _beam_coord.scaling[beam_id] = json.at("scaling").get<int>();
        } catch (std::exception const& e) {
            WARN("[TRACKING] Pointing update failed to read scaling factor from beam: {:d}. {:s}",
                 beam_id, e.what());
            return false;
        }
        INFO("[tracking] Updated Beam={:d} RA={:.2f} Dec={:.2f} Scl={:d}", beam_id,
             _beam_coord.right_ascension[beam_id], _beam_coord.declination[beam_id], _beam_coord.scaling[beam_id]);
    }
    return true;
}

bool BeamformingPhaseUpdate::update_DUT1(nlohmann::json& json) {
    std::lock_guard<std::mutex> lock(beam_lock);
    try {
        _DUT1 = json.at("DUT1").get<double>();
    } catch (std::exception& e) {
        WARN("[FRB] Failed to read DUT1 {:s}", e.what());
        return false;
    }

    return true;
}

BeamformingPhaseUpdate::~BeamformingPhaseUpdate() {}

void BeamformingPhaseUpdate::copy_scaling(const chordMetadata::beamCoord& beam_coord, float* scaling) {
    for (uint32_t i = 0; i < _num_beams; ++i) {
        // In the case that we set scaling to one, we don't want to add the extra factor of
        // 0.5 to the scaling factor, since in the case of scaling == 1, we want no scaling at all.
        if (beam_coord.scaling[i] == 1) {
            scaling[i] = 1;
        } else {
            // @todo Adding 0.5 to the scaling here is hiding the effect from the API
            //       ideally we should just expose the value as a float, and require the user to
            //       apply this extra 0.5 correction directly to their required scaling.
            scaling[i] = beam_coord.scaling[i] + 0.5;
        }
    }
}

void BeamformingPhaseUpdate::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);
    // ID for the frame we are currently using, which we hold onto until there are new gains.
    frameID gains_frame_current_id(gains_buf);
    // ID for the next frame in the gains buffer, used to check for a new frame.
    frameID gains_frame_next_id(gains_buf);
    uint8_t* gains_frame = nullptr;
    bool first_time = true;

    while (!stop_thread) {

        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;
        uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            break;

        float* scaling_frame = (float*)out_frame + _num_elements * _num_beams * 2;

        // If we have a gains buffer we check for new gains
        if (gains_buf != nullptr) {
            if (first_time) {
                gains_frame = gains_buf->wait_for_full_frame(unique_name, gains_frame_current_id);
                // We always get the first frame at startup, so start looking for updates at +1
                gains_frame_next_id++;
                if (gains_frame == nullptr)
                    break;
                first_time = false;
            } else {
                auto timeout = double_to_ts(0);
                // Check if there is a new frame of gains in buffer, if so use it,
                // if not continute using the existing frame.
                auto status = gains_buf->wait_for_full_frame_timeout(unique_name,
                                                          gains_frame_next_id, timeout);
                if (status == -1) {
                    break;
                }
                // There is a new frame of gains so switch to using it.
                if (status == 0) {
                    // Release the current frame we are using.
                    gains_buf->mark_frame_empty(unique_name, gains_frame_current_id++);
                    // The current frame should now be the same as the next frame we will get below.
                    assert(gains_frame_current_id == gains_frame_next_id);
                    // Update to the new frame.
                    gains_frame =
                        gains_buf->wait_for_full_frame(unique_name, gains_frame_next_id++);
                    // In case we exit between the timeout wait, and the wait to get the frame pointer.
                    if (gains_frame == nullptr)
                        break;
                }
            }
        }

        const auto in_meta = get_chord_metadata(in_buf, in_frame_id);

        // Get time
        timespec gps_time = in_meta->get_gps_time();

        // Get frequencies
        const auto coarse_freq = in_meta->get_coarse_freq();
        for (uint32_t i = 0; i < _num_local_freq; ++i) {
            freq_id_t freq_bin = coarse_freq.at(i);
            frequencies_in_frame[i] = Telescope::instance().to_freq_MHz(freq_bin);
        }

        // Lock gain and metadata setting from updating while phase is set and generated
        {
            std::lock_guard<std::mutex> lock(beam_lock);

            // Set the metadata in the input frame to match what pointings we are
            // using at the time the phases are generated
            if (_beam_offset == 0) { // Only update the metadata from one thread
                out_buf->allocate_new_metadata_object(out_frame_id);
                auto out_meta = get_chord_metadata(out_buf, out_frame_id);
                out_meta->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());
                out_meta->set_coarse_freq(coarse_freq);
                out_meta->set_beam_coord(_beam_coord);
            }

            compute_phases(out_frame, gps_time, frequencies_in_frame, _beam_offset, gains_frame);

            if (_beam_offset == 0) // Only update the scaling from one thread
                copy_scaling(_beam_coord, scaling_frame);
        }

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
