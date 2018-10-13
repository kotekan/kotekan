#include "ringmap.hpp"
#include "visBuffer.hpp"
#include <complex>

using namespace std::complex_literals;
const float pi = std::acos(-1);

REGISTER_KOTEKAN_PROCESS(mapMaker);

mapMaker::mapMaker(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&mapMaker::main_thread, this)) {

    // Register REST callback
    using namespace std::placeholders;
    restServer::instance().register_post_callback("ringmap",
        std::bind(&mapMaker::rest_callback,
                this, std::placeholders::_1, std::placeholders::_2
        )
    );

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    freq_id = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");

    if (config.exists(unique_name, "exclude_inputs")) {
        excl_input = config.get<std::vector<uint32_t>>(unique_name,
                                                    "exclude_inputs");
    }

}

void mapMaker::main_thread() {

    if (!setup())
        return;

    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // multiply visibilities with transfer matrix

        // multiply weights

    }
}

nlohmann::json mapMaker::rest_callback(connectionInstance& conn, nlohmann::json& json) {
    // return the map for the specified frequency and polarization in JSON format
    // make sure to lock the map arrays
}

bool mapMaker::setup() {

    // Wait for the input buffer to be filled with data
    if(wait_for_full_frame(in_buf, unique_name.c_str(), 0) == nullptr)
        return false;

    // read products and frequencies from dataset manager

    num_pix = 512; // # unique NS baselines
    num_time = 24. * 360. / 10.; // TODO: can I get integration time from frame?
    num_pol = 4;
    sinza = std::vector<float>(num_pix, 0.);
    for (uint i = 0; i < num_pix; i++) {
        sinza[i] = i * 2. / num_pix - 1. + 1. / num_pix;
    }

    // generate operators for map making
    gen_baselines();
    gen_matrices();

    // initialize map containers
    for (uint p = 0; p < num_pol; p++) {
        map[p].reserve(num_pix*num_time);
        wgt_map[p].reserve(num_pix*num_time);
    }
}

void mapMaker::gen_matrices() {

    // TODO: should the matrix also stack visibilities or do it separately?
    uint32_t n_unique_bl = 256 + 3*511;
    std::vector<float> cyl_stacker = std::vector<float>(num_stack * n_unique_bl, 0.);

    // Map making matrix for each frequency operates on vector of unique baseline visibilities
    for (auto fid : freq_id) {
        std::vector<cfloat> m = vis2map[fid];
        m.reserve(num_time * num_pix * n_unique_bl);
        for (uint t = 0; t < num_time; t++) {
            for (uint p = 0; p < num_pix; p ++) {
                for (uint b = 0; b < n_unique_bl; b ++) {
                    m[(t*num_pix + p)*n_unique_bl + b] = std::exp(cfloat(-2.i) * pi * baselines[b] / wl(fid) * sinza[b]);
                }
            }
        }
    }
}

void mapMaker::gen_baselines() {

    // calculate baseline for every product
}