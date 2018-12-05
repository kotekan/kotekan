#include "ringmap.hpp"
#include "visBuffer.hpp"
#include <complex>
#include <cblas.h>

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

    frameID in_frame_id(in_buf);

    if (!setup(in_frame_id))
        return;

    // coefficients of CBLAS multiplication
    float alpha = 1.;
    float beta = 0.;

    // Initialize the time indexing
    max_fpga, min_fpga = 0;
    latest = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               in_frame_id) == nullptr) {
            break;
        }

        // Check dataset id hasn't changed

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, in_frame_id);
        uint32_t f_id = input_frame.freq_id;

        // Find the time index to append to
        time_ctype t = {std::get<0>(input_frame.time),
                        ts_to_double(std::get<1>(input_frame.time))};
        int64_t t_ind = resolve_time(t);
        if (t_ind >= 0) {
            for (uint p = 0; p < num_pol; p++) {
                // transform into map slice
                cblas_cgemv(CblasRowMajor, CblasNoTrans, num_pix, num_bl,
                            &alpha, vis2map.at(f_id).data, num_pix, &input_frame.vis[p*num_bl],
                            1, &beta, &map.at(f_id).at(p).data[t_ind*num_pix], 1);

                 // same for weights map
                cblas_cgemv(CblasRowMajor, CblasNoTrans, num_pix, num_bl,
                            &alpha, vis2map.at(f_id).data, num_pix, &input_frame.weight[p*num_bl],
                            1, &beta, &wgt_map.at(f_id).at(p).data[t_ind*num_pix], 1);
            }
        }
        // Move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
    }
}

nlohmann::json mapMaker::rest_callback(connectionInstance& conn, nlohmann::json& json) {
    // return the map for the specified frequency and polarization in JSON format
    // make sure to lock the map arrays
}

bool mapMaker::setup(size_t frame_id) {

    // Wait for the input buffer to be filled with data
    if(wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr)
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
    for (auto fid : freq_id) {
        map.at(fid).reserve(num_pol);
        wgt_map.at(fid).reserve(num_pol);
        for (uint p = 0; p < num_pol; p++) {
            map.at(fid).at(p).reserve(num_pix*num_time);
            wgt_map.at(fid).at(p).reserve(num_pix*num_time);
        }
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

int64_t mapMaker::resolve_time(time_ctype t){

    if (t.fpga_count < min_fpga) {
        // time is too old, discard
        WARN("Frame older than oldest time in ringmap. Discarding.");
        return -1;
    }

    if (t.fpga_count > max_fpga) {
        // We need to add a new time
        max_fpga = t.fpga_count;
        // Increment position and remove previous entry
        min_fpga = times[latest++].fpga_count;
        times_map.erase(min_fpga);
        for (auto fid : freq_id) {
            for (uint p = 0; p < num_pol; p++) {
                std::fill(map.at(fid).at(p).begin() + latest*num_pix,
                          map.at(fid).at(p).begin() + (latest+1)*num_pix, 0.);
                std::fill(wgt_map.at(fid).at(p).begin() + latest*num_pix,
                          wgt_map.at(fid).at(p).begin() + (latest+1)*num_pix, 0.);
        }
        times[latest] = t;
        times_map.at(t.fpga_count) = latest;

        return latest;
    }

    // Otherwise find the existing time
    auto res = times_map.find(t.fpga_count);
    if (res == times_map.end()) {
        // No entry for this time
        WARN("Could not find this time in ringmap. Discarding.");
        return -1;
    }
    return res->second;
}