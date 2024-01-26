#include "BeamformingPhaseUpdate.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"         // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "chimeMetadata.hpp"
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG
#include "util.h"             // for hex_dump


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

STAGE_CONSTRUCTOR(BeamformingPhaseUpdate) {

    // Register on buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    if (config.exists(unique_name.c_str(), "gains_buf")) {
        gains_buf = get_buffer("gains_buf");
        register_consumer(gains_buf, unique_name.c_str());
    }

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_beams = config.get<int16_t>(unique_name, "num_beams");

    _inst_lat = config.get<double>(unique_name, "inst_lat");
    _inst_long = config.get<double>(unique_name, "inst_long");

    _num_local_freq = config.get<double>(unique_name, "num_local_freq");

    frequencies_in_frame.resize(_num_local_freq);

    // Get feed locations
    feed_locations =
        config.get<std::vector<std::pair<double, double>>>(unique_name, "feed_positions");

    // listen for UT1_UTC offset
    std::string UT1_UTC =
        config.get_default<std::string>(unique_name, "updatable_config/UT1_UTC", "");
    if (UT1_UTC.length() > 0)
        kotekan::configUpdater::instance().subscribe(
            UT1_UTC, std::bind(&BeamformingPhaseUpdate::update_UT1_UTC_offset, this, std::placeholders::_1));

    // Register function to listen for new beam, and update ra and dec
    using namespace std::placeholders;
    for (int beam_id = 0; beam_id < _num_beams; beam_id++) {
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
            _beam_coord.ra[beam_id] = json.at("ra").get<float>();
        } catch (std::exception const& e) {
            WARN("[TRACKING] Pointing update failed to read RA from beam: {:d}. {:s}", beam_id,
                 e.what());
            return false;
        }
        try {
            _beam_coord.dec[beam_id] = json.at("dec").get<float>();
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
             _beam_coord.ra[beam_id], _beam_coord.dec[beam_id], _beam_coord.scaling[beam_id]);
    }
    return true;
}

bool BeamformingPhaseUpdate::update_UT1_UTC_offset(nlohmann::json& json) {
    std::lock_guard<std::mutex> lock(beam_lock);
    try {
        _DUT1 = json.at("UT1_UTC_val").get<double>();
    } catch (std::exception& e) {
        WARN("[FRB] Fail to read UT1_UTC offset {:s}", e.what());
        return false;
    }

    return true;
}

BeamformingPhaseUpdate::~BeamformingPhaseUpdate() {}

void BeamformingPhaseUpdate::copy_scaling(const beamCoord& beam_coord, float* scaling) {
    for (int i = 0; i < _num_beams; ++i) {
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
    frameID gains_frame_id(gains_buf);
    uint8_t* gains_frame = nullptr;
    bool first_time = true;

    while (!stop_thread) {

        uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;
        uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
        if (out_frame == nullptr)
            break;

        float* scaling_frame = (float*)out_frame + _num_elements * _num_beams * 2;

        // If we have a gains buffer we check for new gains
        if (gains_buf != nullptr) {
            if (first_time) {
                gains_frame = wait_for_full_frame(gains_buf, unique_name.c_str(), gains_frame_id++);
                if (gains_frame == nullptr)
                    break;
                first_time = false;
            } else {
                auto timeout = double_to_ts(0);
                auto status = wait_for_full_frame_timeout(gains_buf, unique_name.c_str(),
                                                          gains_frame_id, timeout);
                if (status == -1) {
                    break;
                }
                if (status == 0) {
                    mark_frame_empty(gains_buf, unique_name.c_str(), gains_frame_id++);
                    gains_frame =
                        wait_for_full_frame(gains_buf, unique_name.c_str(), gains_frame_id);
                }
            }
        }

        // Get time
        timespec gps_time = get_gps_time(in_buf, in_frame_id);

        // Get frequencies
        for (uint32_t i = 0; i < _num_local_freq; ++i) {
            frequencies_in_frame[i] = Telescope::instance().to_freq_id(in_buf, in_frame_id, i);
            // Get frequency in MHz with Telescope::instance().to_freq(frequencies_in_frame[i])
        }

        // Lock gain and metadata setting from updating while phase is set and generated
        {
            std::lock_guard<std::mutex> lock(beam_lock);
            // Set the metadata in the input frame to match what pointings we are
            // using at the time the phases are generated
            set_beam_coord(in_buf, in_frame_id, _beam_coord);
            compute_phases(out_frame, gps_time, frequencies_in_frame, gains_frame);
            copy_scaling(_beam_coord, scaling_frame);
	    //for (int i = 0; i < out_buf->frame_size/sizeof(float); i += 1) {
                //INFO("phases[{:d} = {:f}", i, ((float*)out_frame)[i]);
	    //}
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
    }
}
