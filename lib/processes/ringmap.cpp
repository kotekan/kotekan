#include "ringmap.hpp"
#include "visBuffer.hpp"
#include "datasetManager.hpp"
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

    if (config.exists(unique_name, "exclude_inputs")) {
        excl_input = config.get<std::vector<uint32_t>>(unique_name,
                                                    "exclude_inputs");
    }

}

void mapMaker::main_thread() {

    if (!setup(0))
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

bool mapMaker::setup(uint frame_id) {

    // Wait for the input buffer to be filled with data
    if(wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr)
        return false;

    auto frame = visFrameView(in_buf, frame_id);
    ds_id = frame.dataset_id;

    change_dataset_state();

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

void mapMaker::change_dataset_state() {

    auto& dm = datasetManager::instance();

    // Get the frequency spec to determine the freq_ids expected at this Writer.
    auto fstate = dm.dataset_state<freqState>(ds_id);
    if (fstate == nullptr)
        throw std::runtime_error("Could not find freqState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");

    freq = fstate->get_freqs();

    // Get the product spec and (if available) the stackState to determine the
    // number of vis entries we are expecting
    auto pstate = dm.dataset_state<prodState>(ds_id);
    auto sstate = dm.dataset_state<stackState>(ds_id);
    auto mstate = dm.dataset_state<metadataState>(ds_id);
    if (pstate == nullptr || sstate == nullptr || mstate == nullptr)
        throw std::runtime_error("Could not find all dataset states for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".\n" \
                                 "One of them is a nullptr (0): prod "
                                 + std::to_string(pstate != nullptr)
                                 + ", stack "
                                 + std::to_string(sstate != nullptr)
                                 + ", metadata "
                                 + std::to_string(mstate != nullptr));

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    //if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
    //    INFO("Git version tags don't match: dataset %zu has tag %s, while "\
    //         "the local git version tag is %s", ds_id,
    //         mstate->get_git_version_tag().c_str(),
    //         get_git_commit_hash());
    //}

    if (!sstate.is_stacked())
        throw std::runtime_error("MapMaker requires visibilities stacked ")
    num_stack = sstate->is_stacked() ?
            sstate->get_num_stack() : pstate->get_prods().size();
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