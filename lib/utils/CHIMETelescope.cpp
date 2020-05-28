#include "CHIMETelescope.hpp"

#include "ICETelescope.hpp"
#include "Telescope.hpp"


REGISTER_TELESCOPE(CHIMETelescope, "chime");


CHIMETelescope::CHIMETelescope(const kotekan::Config& config, const std::string& path) :
    ICETelescope(config.get<std::string>(path, "log_level")) {

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream = config.get<uint32_t>(path, "num_local_freq");

    set_sampling_params(800.0, 2048, 2);
    set_gps(config, path);
    set_frequency_map(config, "fpga_frequency_map");
}

void CHIMETelescope::set_frequency_map(const kotekan::Config& config, const std::string& path) {
    nlohmann::json fpga_freq_map_json = config.get_default<nlohmann::json>("/", path, {});


    if (fpga_freq_map_json.empty()) {
        WARN("No frequency map provided, generating default map");

        // Default FPGA mapping will be incorrect for any dynamic map
        // This function generates the default FPGA table mapping in the CHIME
        // configuration 4 pairs of ICEBoards, won't work for any other config.
        for (uint32_t i = 0; i < nfreq; ++i) {
            ice_stream_id_t stream_id = {0, 0, 0, 0};

            frequency_table[ice_encode_stream_id(stream_id).id] = i;

            stream_id.slot_id = (stream_id.slot_id + 1) % 16;

            // Only update slot_id on first pass.
            if (i == 0)
                continue;

            if (i % 256 == 0) {
                stream_id.unused = (stream_id.unused + 1) % 4;
            }
            if (i % 32 == 0) {
                stream_id.link_id = (stream_id.link_id + 1) % 8;
            }
            if (i % 16 == 0) {
                stream_id.crate_id = (stream_id.crate_id + 1) % 2;
            }
        }
    } else {
        // Extract the freq map
        // TODO put this in a try catch
        std::map<std::string, std::vector<int>> fpga_freq_map =
            fpga_freq_map_json.at("fmap").get<std::map<std::string, std::vector<int>>>();

        // We do not use the FPGA table directly because we do a final corner-turn
        // to generate streams of data with 1 frequency per-stream instead of the
        // 4 frequencies per-stream the FPGAs send on each of the 4 links.
        ice_stream_id_t fpga_stream_id = {0, 0, 0, 0};
        ice_stream_id_t post_shuffle_id = {0, 0, 0, 0};
        for (auto& [fpga_encoded_stream_id, bin_numbers] : fpga_freq_map) {
            int index = 0;
            for (auto& bin : bin_numbers) {
                fpga_stream_id = ice_extract_stream_id({std::stoull(fpga_encoded_stream_id)});

                // We assume that all 4 crate pairs are generating the same map
                // TODO should we check this?
                if (fpga_stream_id.crate_id != 0 && fpga_stream_id.crate_id != 1)
                    break;

                post_shuffle_id.crate_id = fpga_stream_id.crate_id;
                post_shuffle_id.slot_id = fpga_stream_id.slot_id;
                post_shuffle_id.link_id = fpga_stream_id.link_id;
                post_shuffle_id.unused = index++;


                frequency_table[ice_encode_stream_id(post_shuffle_id).id] = bin;
            }
        }
    }
}

freq_id_t CHIMETelescope::to_freq_id(stream_t stream, uint32_t /* ind */) const {
    return frequency_table.at(stream.id);
}