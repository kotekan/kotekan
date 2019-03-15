#include "ringmap.hpp"
#include "visBuffer.hpp"
#include "datasetManager.hpp"
#include "StageFactory.hpp"
#include "visCompression.hpp"
#include <complex>
#include <cblas.h>

using namespace std::complex_literals;
using namespace std::placeholders;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::prometheusMetrics;
using kotekan::Stage;
using kotekan::restServer;
using kotekan::HTTP_RESPONSE;

const float pi = std::acos(-1);

REGISTER_KOTEKAN_STAGE(mapMaker);
REGISTER_KOTEKAN_STAGE(redundantStack);

mapMaker::mapMaker(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&mapMaker::main_thread, this)) {

    // Register REST callbacks
    restServer::instance().register_post_callback("ringmap",
        std::bind(&mapMaker::rest_callback,
                  this, std::placeholders::_1, std::placeholders::_2
        )
    );
    restServer::instance().register_get_callback("ringmap",
        std::bind(&mapMaker::rest_callback_get,
                  this, std::placeholders::_1
        )
    );

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // config parameters
    feed_sep = config.get_default<float>(unique_name, "feed_sep", 0.3048);

}

void mapMaker::main_thread() {

    // coefficients of CBLAS multiplication
    float alpha = 1.;
    float beta = 0.;

    frameID in_frame_id(in_buf);

    if (!setup(in_frame_id))
        return;

    // These will be used to get around the missing cross-polar visibility
    // TODO: this is not at all generic
    size_t offset;
    uint p_special = 2;  // This is the pol that is shorter than the others
    std::vector<cfloat> special_vis(num_bl);
    std::vector<cfloat> special_wgt(num_bl);
    // We will need to cast weights into complex
    std::vector<cfloat> complex_wgt(num_stack);
    // Buffers to hold result before saving real part
    std::vector<cfloat> tmp_vismap(num_pix);
    std::vector<cfloat> tmp_wgtmap(num_pix);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               in_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, in_frame_id);
        uint32_t f_id = input_frame.freq_id;

        // Check dataset id hasn't changed
        if (input_frame.dataset_id != ds_id) {
            size_t old_num_stack = num_stack;
            change_dataset_state(input_frame.dataset_id);
            // This should never happen...
            if (num_stack != old_num_stack) {
                // Need to regenerate matrices
                if (!setup(in_frame_id))
                    return;
            }
        }

        // Find the time index to append to
        time_ctype t = {std::get<0>(input_frame.time),
                        ts_to_double(std::get<1>(input_frame.time))};
        int64_t t_ind = resolve_time(t);
        if (t_ind >= 0) {
            // Copy weights into a complex vector
            std::transform(
                input_frame.weight.begin(), input_frame.weight.begin() + num_stack,
                complex_wgt.begin(),
                [](const float& a) {return cfloat(a, 0.);}
            );
            mtx.lock();
            for (uint p = 0; p < num_pol; p++) {
                // Pointers to the span of visibilities for this pol
                cfloat* input_vis;
                cfloat* input_wgt;

                if (p != p_special) {
                    // Need offset to account for missing cross-pol
                    offset = p * num_bl - (p > p_special);
                    input_vis = input_frame.vis.data() + offset;
                    input_wgt = complex_wgt.data() + offset;
                } else {
                    // For now just copy the visibility. This might be slow...
                    std::copy(input_frame.vis.begin() + p * num_bl,
                        input_frame.vis.begin() + (p + 1) * num_bl - 1,
                        special_vis.begin() + 1);
                    // for the weights, need to cast
                    std::copy(complex_wgt.begin() + p * num_bl,
                        complex_wgt.begin() + (p + 1) * num_bl - 1,
                        special_wgt.begin() + 1);
                    // Add missing cross-pol
                    special_vis.at(0) = conj(input_frame.vis.at((p - 1) * num_bl));
                    special_wgt.at(0) = complex_wgt.at((p - 1) * num_bl);

                    input_vis = special_vis.data();
                    input_wgt = special_wgt.data();
                }
                // transform into map slice
                cblas_cgemv(CblasRowMajor, CblasNoTrans, num_pix, num_bl,
                            &alpha, vis2map.at(f_id).data(), num_bl, input_vis,
                            1, &beta, map.at(f_id).at(p).data() + t_ind * num_pix, 1);

                // same for weights map
                cblas_cgemv(CblasRowMajor, CblasNoTrans, num_pix, num_bl,
                            &alpha, vis2map.at(f_id).data(), num_bl, input_wgt,
                            1, &beta, wgt_map.at(f_id).at(p).data() + t_ind * num_pix, 1);

                // multiply visibility and weight maps
                // keep real part only
                uint map_offset = t_ind * num_pix;
                for (uint i = 0; i < num_pix; i++) {
                    wgt_map.at(f_id).at(p).at(map_offset + i) = (tmp_vismap.at(i) * tmp_wgtmap.at(i)).real();
                    map.at(f_id).at(p).at(map_offset + i) = tmp_vismap.at(i).real();
                }
            }
            mtx.unlock();
        }
        // Move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
    }
}

void mapMaker::rest_callback_get(kotekan::connectionInstance& conn) {

    // Return the available frequencies and polarisations
    nlohmann::json resp;
    std::vector<float> fphys;
    for (auto f : freqs)
        fphys.push_back(f.second.centre);
    resp["freq"] = nlohmann::json(fphys);
    std::vector<int> pol;
    for (int i = 0; i < num_pol; i++)
        pol.push_back(i);
    resp["pol"] = nlohmann::json(pol);
    conn.send_json_reply(resp);
    return;
}

void mapMaker::rest_callback(kotekan::connectionInstance& conn,
                                       nlohmann::json& json) {
    // return the map for the specified frequency and polarization in JSON format
    // make sure to lock the map arrays

    // Extract requested polarization and frequency
    int pol;
    if (json.find("pol") != json.end()) {
        pol = json.at("pol");
    } else {
        conn.send_error("Did not find key 'pol' in JSON request.",
                       HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int f_ind;
    if (json.find("freq_ind") != json.end()) {
        f_ind = json.at("freq_ind");
    } else {
        conn.send_error("Did not find key 'freq_ind' in JSON request.",
                       HTTP_RESPONSE::BAD_REQUEST);
        return;
    }

    // TODO: Process weights map here to save data transfer

    // Pack map into msgpack
    nlohmann::json resp;
    mtx.lock();
    resp["time"] = nlohmann::json(times);
    resp["sinza"] = nlohmann::json(sinza);
    resp["ringmap"] = nlohmann::json(map.at(freqs[f_ind].first).at(pol));
    resp["weight_map"] = nlohmann::json(wgt_map.at(freqs[f_ind].first).at(pol));
    std::vector<std::uint8_t> resp_msgpack = nlohmann::json::to_msgpack(resp);
    mtx.unlock();
    conn.send_binary_reply(resp_msgpack.data(), resp_msgpack.size());
    return;
}

void mapMaker::change_dataset_state(dset_id_t new_ds_id) {

    // Update stored ID
    ds_id = new_ds_id;

    auto& dm = datasetManager::instance();

    // Get the product spec and (if available) the stackState to determine the
    // number of vis entries we are expecting
    auto istate = dm.dataset_state<inputState>(ds_id);
    auto pstate = dm.dataset_state<prodState>(ds_id);
    auto sstate = dm.dataset_state<stackState>(ds_id);
    auto fstate = dm.dataset_state<freqState>(ds_id);
    if (pstate == nullptr || istate == nullptr || fstate == nullptr)
        throw std::runtime_error("Could not find all dataset states for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".\n" \
                                 "One of them is a nullptr (0): prod "
                                 + std::to_string(pstate != nullptr)
                                 + ", input "
                                 + std::to_string(istate != nullptr)
                                 + ", stack "
                                 + std::to_string(sstate != nullptr));

    if (sstate == nullptr)
        throw std::runtime_error("MapMaker requires visibilities stacked ");

    stacks = sstate->get_stack_map();
    prods = pstate->get_prods();
    inputs = istate->get_inputs();
    freqs = fstate->get_freqs();

    num_stack = sstate->get_num_stack();
}

bool mapMaker::setup(size_t frame_id) {

    // Wait for the input buffer to be filled with data
    if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
        return false;
    }

    auto in_frame = visFrameView(in_buf, frame_id);
    ds_id = in_frame.dataset_id;
    change_dataset_state(ds_id);

    // TODO: make these config options ?
    num_pix = 512; // # unique NS baselines
    num_pol = 4;
    num_time = 24. * 3600. / (in_frame.fpga_seq_length * 2.56e-6);
    num_bl = (num_stack + 1) / 4;

    sinza = std::vector<float>(num_pix, 0.);
    for (uint i = 0; i < num_pix; i++) {
        sinza[i] = i * 2. / num_pix - 1. + 1. / num_pix;
    }

    min_fpga = std::get<0>(in_frame.time);

    // generate map making matrices
    gen_matrices();

    // initialize map containers
    mtx.lock();
    for (auto f : freqs) {
        std::vector<std::vector<float>> vis(num_pol);
        std::vector<std::vector<float>> wgt(num_pol);
        for (uint p = 0; p < num_pol; p++) {
            vis.at(p).resize(num_time * num_pix);
            wgt.at(p).resize(num_time * num_pix);
            std::fill(vis.at(p).begin(), vis.at(p).end(), 0.);
            std::fill(wgt.at(p).begin(), wgt.at(p).end(), 0.);
        }
        map.insert(std::pair<uint64_t, std::vector<std::vector<float>>>(f.first, vis));
        wgt_map.insert(std::pair<uint64_t, std::vector<std::vector<float>>>(f.first, wgt));
    }
    mtx.unlock();

    // Make sure times are empty
    times.clear();
    times_map.clear();

    // Initialize the time indexing
    max_fpga = 0, min_fpga = 0;
    latest = modulo<size_t>(num_time);

    return true;
}

void mapMaker::gen_matrices() {

    // calculate baseline for every stacked product
    ns_baselines.reserve(num_bl);
    chimeFeed input_a, input_b;
    for (size_t i = 0; i < num_bl; i++) {
        stack_ctype s = stacks[i];
        input_a = chimeFeed::from_input(inputs[prods[s.prod].input_a]);
        input_b = chimeFeed::from_input(inputs[prods[s.prod].input_b]);
        ns_baselines[i] = feed_sep * (input_b.feed_location - input_a.feed_location);
        if (s.conjugate)
            ns_baselines[i] *= -1;
    }

    // Construct matrix of phase weights for every baseline and pixel
    for (auto f : freqs) {
        std::vector<cfloat> m(num_pix * num_bl);
        float lam = wl(f.second.centre);
        for (uint p = 0; p < num_pix; p++) {
            for (uint i = 0; i < num_bl; i++) {
                m[p*num_bl + i] = std::exp(cfloat(-2.i) * pi * ns_baselines[i] / lam * sinza[p]);
            }
        }
        vis2map.insert(std::pair<uint64_t, std::vector<cfloat>>(f.first, m));
    }
}

int64_t mapMaker::resolve_time(time_ctype t){

    if (t.fpga_count < min_fpga) {
        // time is too old, discard
        WARN("Frame older than oldest time in ringmap. Discarding.");
        return -1;
    }

    if (t.fpga_count > max_fpga) {
        mtx.lock();
        // We need to add a new time
        max_fpga = t.fpga_count;
        // Increment position
        if (times_map.size() < num_time) {
            // Still filling in the array
            if (times_map.size() > 0)
                latest++;
            times.push_back(t);
        } else {
            // Remove oldest entry
            min_fpga = times[++latest].fpga_count;
            times_map.erase(min_fpga);
            times[latest] = t;
        }
        times_map.insert(std::pair<uint64_t, size_t>(t.fpga_count, latest));
        // Clear maps
        size_t start = latest;
        size_t stop = size_t(latest) + 1;
        for (auto f : freqs) {
            uint64_t fid = f.first;
            for (uint p = 0; p < num_pol; p++) {
                std::fill(map.at(fid).at(p).begin() + start*num_pix,
                          map.at(fid).at(p).begin() + stop*num_pix, cfloat(0., 0.));
                std::fill(wgt_map.at(fid).at(p).begin() + start*num_pix,
                          wgt_map.at(fid).at(p).begin() + stop*num_pix, cfloat(0., 0.));
            }
        }
        mtx.unlock();

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

redundantStack::redundantStack(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&redundantStack::main_thread, this)) {

    // Get buffers from config
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

}

void redundantStack::change_dataset_state(dset_id_t ds_id) {
    auto& dm = datasetManager::instance();
    state_id_t stack_state_id;

    // Get input & prod states synchronoulsy
    const inputState*  input_state_ptr = dm.dataset_state<inputState>(ds_id);
    prod_state_ptr = dm.dataset_state<prodState>(ds_id);
    old_stack_state_ptr = dm.dataset_state<stackState>(ds_id);

    if (input_state_ptr == nullptr)
        throw std::runtime_error("Could not find inputState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");
    if (prod_state_ptr == nullptr)
        throw std::runtime_error("Could not find prodState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");
    if (old_stack_state_ptr == nullptr)
        throw std::runtime_error("Could not find stackState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");

    auto sspec = full_redundant(input_state_ptr->get_inputs(),
                                prod_state_ptr->get_prods());
    auto sstate = std::make_unique<stackState>(
        sspec.first, std::move(sspec.second));

    std::tie(stack_state_id, new_stack_state_ptr) =
        dm.add_state(std::move(sstate));

    output_dset_id = dm.add_dataset(ds_id, stack_state_id);
}

void redundantStack::main_thread() {

    frameID in_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    // Wait for the input buffer to be filled with data
    if(wait_for_full_frame(in_buf, unique_name.c_str(),
                           in_frame_id) == nullptr) {
        return;
    }

    auto input_frame = visFrameView(in_buf, in_frame_id);
    input_dset_id = input_frame.dataset_id;
    change_dataset_state(input_dset_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               in_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, in_frame_id);

        // Check dataset id hasn't changed
        if (input_frame.dataset_id != input_dset_id) {
            WARN("Input dataset ID has changed. Regenerating stack specs.");
            input_dset_id = input_frame.dataset_id;
            change_dataset_state(input_dset_id);
        }

        const auto& stack_rmap = new_stack_state_ptr->get_rstack_map();
        const auto& old_stack_map = old_stack_state_ptr->get_stack_map();
        const auto& prods = prod_state_ptr->get_prods();
        auto num_stack = new_stack_state_ptr->get_num_stack();

        std::vector<float> stack_norm(new_stack_state_ptr->get_num_stack(), 0.0);
        std::vector<float> stack_v2(new_stack_state_ptr->get_num_stack(), 0.0);

        // Wait for the output buffer frame to be free
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);
        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         input_frame.num_elements,
                                         num_stack,
                                         input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {visField::vis, visField::weight});
        output_frame.dataset_id = output_dset_id;

        // Zero the output frame
        std::fill(std::begin(output_frame.vis),
                  std::end(output_frame.vis), 0.0);
        std::fill(std::begin(output_frame.weight),
                  std::end(output_frame.weight), 0.0);

        auto in_vis = input_frame.vis.data();
        auto out_vis = output_frame.vis.data();
        auto in_weight = input_frame.weight.data();
        auto out_weight = output_frame.weight.data();
        auto flags = output_frame.flags.data();

        // Iterate over all the products and average together
        for(uint32_t old_ind = 0; old_ind < old_stack_map.size(); old_ind++) {
            // TODO: if the weights are ever different from 0 or 1, we will
            // definitely need to rewrite this.

            // Alias the parts of the data we are going to stack
            cfloat vis = in_vis[old_ind];
            float weight = in_weight[old_ind];

            auto& old_s = old_stack_map[old_ind];
            auto& p = prods[old_s.prod];
            auto& s = stack_rmap[old_s.prod];

            // If the weight is zero, completey skip this iteration
            if (weight == 0 || flags[p.input_a] == 0 || flags[p.input_b] == 0)
                continue;

            vis = (s.conjugate != old_s.conjugate) ? conj(vis) : vis;

            // First summation of the visibilities (dividing by the total weight will be done later)
            out_vis[s.stack] += vis;

            // Accumulate the square for variance calculation
            stack_v2[s.stack] += fast_norm(vis);

            // Accumulate the weighted *variances*. Normalising and inversion
            // will be done later
            out_weight[s.stack] += (1.0 / weight);

            // Accumulate the weights so we can normalize correctly
            stack_norm[s.stack] += 1.0;
        }

        // Loop over the stacks and normalise (and invert the variances)
        float vart = 0.0;
        float normt = 0.0;
        for(uint32_t stack_ind = 0; stack_ind < num_stack; stack_ind++) {

            // Calculate the mean and accumulate weight and place in the frame
            float norm = stack_norm[stack_ind];

            // Invert norm if set, otherwise use zero to set data to zero.
            float inorm = (norm != 0.0) ? (1.0 / norm) : 0.0;

            output_frame.vis[stack_ind] *= inorm;
            float iwgt = ((output_frame.weight[stack_ind] != 0.0)
                          ? 1.0 / output_frame.weight[stack_ind] : 0.0);
            output_frame.weight[stack_ind] = norm * norm * iwgt;

            // Accumulate to calculate the variance of the residuals
            vart += stack_v2[stack_ind]
                    - std::norm(output_frame.vis[stack_ind]) * norm;
            normt += norm;
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
    }
}

using feed_diff = std::tuple<int8_t, int8_t, int8_t, int16_t>;

// Calculate the baseline parameters and whether the product must be
// conjugated to get canonical ordering
// Modified to return cylinder separation
std::pair<feed_diff, bool> calculate_chime_vis_full(
    const prod_ctype& p, const std::vector<input_ctype>& inputs)
{

    chimeFeed fa = chimeFeed::from_input(inputs[p.input_a]);
    chimeFeed fb = chimeFeed::from_input(inputs[p.input_b]);

    bool is_wrong_cylorder = (fa.cylinder > fb.cylinder);
    bool is_same_cyl_wrong_feed_order = (
            (fa.cylinder == fb.cylinder) &&
            (fa.feed_location > fb.feed_location)
    );
    bool is_same_feed_wrong_pol_order = (
        (fa.cylinder == fb.cylinder) &&
        (fa.feed_location == fb.feed_location) &&
        (fa.polarisation > fb.polarisation)
    );

    bool conjugate = false;

    // Check if we need to conjugate/transpose to get the correct order
    if (is_wrong_cylorder ||
        is_same_cyl_wrong_feed_order ||
        is_same_feed_wrong_pol_order) {

        chimeFeed t = fa;
        fa = fb;
        fb = t;
        conjugate = true;
    }

    return {
        std::make_tuple(fa.polarisation, fb.polarisation, fb.cylinder - fa.cylinder,
                        fb.feed_location - fa.feed_location),
        conjugate
    };
}

// Only modification is to use fully redundant stacking
std::pair<uint32_t, std::vector<rstack_ctype>>
full_redundant(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods) {
    // Calculate the set of baseline properties
    std::vector<std::pair<feed_diff, bool>> bl_prop;
    std::transform(std::begin(prods), std::end(prods), std::back_inserter(bl_prop),
                   std::bind(calculate_chime_vis_full, _1, inputs));

    // Create an index array for doing the sorting
    std::vector<uint32_t> sort_ind(prods.size());
    std::iota(std::begin(sort_ind), std::end(sort_ind), 0);

    auto sort_fn = [&](const uint32_t& ii, const uint32_t& jj) -> bool {
        return (bl_prop[ii].first < bl_prop[jj].first);
    };
    std::sort(std::begin(sort_ind), std::end(sort_ind), sort_fn);

    std::vector<rstack_ctype> stack_map(prods.size());

    feed_diff cur = bl_prop[sort_ind[0]].first;
    uint32_t cur_stack_ind = 0;

    for (auto& ind : sort_ind) {
        if (bl_prop[ind].first != cur) {
            cur = bl_prop[ind].first;
            cur_stack_ind++;
        }
        stack_map[ind] = {cur_stack_ind, bl_prop[ind].second};
    }

    return {++cur_stack_ind, stack_map};
}
