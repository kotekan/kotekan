#include "RingMapMaker.hpp"

#include "Hash.hpp"         // for Hash, operator!=
#include "Stack.hpp"        // for chimeFeed
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "datasetManager.hpp"    // for datasetManager, dset_id_t, state_id_t
#include "kotekanLogging.hpp"    // for FATAL_ERROR, WARN
#include "prometheusMetrics.hpp" // for Metrics
#include "visBuffer.hpp"         // for VisFrameView, VisField, VisField::vis, VisField::weight
#include "visCompression.hpp"    // for chimeFeed

#include "gsl-lite.hpp" // for span, span<>::iterator

#include <atomic>       // for atomic_bool
#include <cblas.h>      // for cblas_cgemv, CblasNoTrans, CblasRowMajor
#include <complex>      // for operator*, complex, operator/, norm, operator-, operato...
#include <cstdint>      // for uint32_t, uint64_t, int64_t, uint8_t, int16_t
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, _Placeholder, bind, _1, function, _2
#include <future>       // for async, future
#include <iterator>     // for begin, end, back_insert_iterator, back_inserter
#include <memory>       // for allocator_traits<>::value_type, make_unique
#include <numeric>      // for iota
#include <regex>        // for match_results<>::_Base_type
#include <sys/types.h>  // for uint, int8_t
#include <system_error> // for system_error
#include <tuple>        // for get, tuple, make_tuple, operator!=, operator<, tie

using namespace std::complex_literals;
using namespace std::placeholders;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(RingMapMaker);
REGISTER_KOTEKAN_STAGE(RedundantStack);

RingMapMaker::RingMapMaker(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RingMapMaker::main_thread, this)) {

    // Register REST callbacks
    restServer::instance().register_post_callback("ringmap", std::bind(&RingMapMaker::rest_callback,
                                                                       this, std::placeholders::_1,
                                                                       std::placeholders::_2));
    restServer::instance().register_get_callback(
        "ringmap", std::bind(&RingMapMaker::rest_callback_get, this, std::placeholders::_1));

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // config parameters
    feed_sep = config.get_default<float>(unique_name, "feed_sep", 0.3048);
    apodization = config.get_default<std::string>(unique_name, "apodization", "nuttall");
    if (apod_param.count(apodization) == 0)
        FATAL_ERROR("Unknown apodization window '{}'", apodization);
    exclude_autos = config.get_default<bool>(unique_name, "exclude_autos", true);
}

void RingMapMaker::main_thread() {

    // coefficients of CBLAS multiplication
    float alpha = 1.;
    float beta = 0.;

    frameID in_frame_id(in_buf);

    if (!setup(in_frame_id))
        return;

    // These will be used to get around the missing cross-polar visibility
    // TODO: this is not at all generic
    size_t offset;
    uint p_special = 2; // This is the pol that is shorter than the others
    std::vector<cfloat> special_vis(num_bl);
    // We will need to cast weights into complex
    std::vector<float> frame_var(num_stack);
    // Buffers to hold result before saving real part
    std::vector<cfloat> tmp_vismap(num_pix);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = VisFrameView(in_buf, in_frame_id);
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
        time_ctype t = {std::get<0>(input_frame.time), ts_to_double(std::get<1>(input_frame.time))};
        int64_t t_ind = resolve_time(t);
        if (t_ind >= 0) {
            // Copy variances into a vector
            std::transform(input_frame.weight.begin(), input_frame.weight.begin() + num_stack,
                           frame_var.begin(),
                           [](const float& a) { return (a != 0.) ? 1. / a : 0.; });
            mtx.lock();
            for (uint p = 0; p < num_pol; p++) {
                // Pointer to the span of visibilities for this pol
                cfloat* input_vis;

                // Sum of variances for this pol
                float var = 0.;

                offset = p * num_bl;
                if (p != p_special) {
                    // Need offset to account for missing cross-pol
                    offset -= (p > p_special);
                    input_vis = input_frame.vis.data() + offset;
                } else {
                    // For now just copy the visibility. This might be slow...
                    std::copy(input_frame.vis.begin() + p * num_bl,
                              input_frame.vis.begin() + (p + 1) * num_bl - 1,
                              special_vis.begin() + 1);
                    // Add missing cross-pol
                    special_vis.at(0) = conj(input_frame.vis.at((p - 1) * num_bl));
                    var = frame_var.at((p - 1) * num_bl);

                    input_vis = special_vis.data();
                }
                // transform into map slice
                std::fill(tmp_vismap.begin(), tmp_vismap.end(), cfloat(0., 0.));

                // NOTE: in here we explicitly cast down to float, to fit the API of old versions of
                // OpenBLAS which require (float *) for complex arrays. Newer versions just accept
                // (void *).
                cblas_cgemv(CblasRowMajor, CblasNoTrans, num_pix, num_bl, &alpha,
                            (float*)vis2map.at(f_id).data(), num_bl, (float*)input_vis, 1, &beta,
                            (float*)tmp_vismap.data(), 1);

                // keep real part only
                for (size_t i = 0; i < num_pix; i++) {
                    map.at(f_id).at(p).at(t_ind * num_pix + i) = tmp_vismap.at(i).real();
                }
                // accumulate variances. for special pol, we already have the first entry
                for (size_t i = 0; i < num_pix - (p == p_special); i++) {
                    var += frame_var.at(offset + i);
                }
                // variance of real part is half, we've divided by number of baselines
                wgt.at(f_id).at(p).at(t_ind) = (var != 0.) ? 2. * num_bl * num_bl / var : 0.;
            }
            mtx.unlock();
        }
        // Move to next frame
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
    }
}

void RingMapMaker::rest_callback_get(kotekan::connectionInstance& conn) {

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

void RingMapMaker::rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json) {
    // return the map for the specified frequency and polarization in JSON format
    // make sure to lock the map arrays

    // Extract requested polarization and frequency
    int pol;
    if (json.find("pol") != json.end()) {
        pol = json.at("pol");
    } else {
        conn.send_error("Did not find key 'pol' in JSON request.", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int f_ind;
    if (json.find("freq_ind") != json.end()) {
        f_ind = json.at("freq_ind");
    } else {
        conn.send_error("Did not find key 'freq_ind' in JSON request.", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }

    // Pack map into msgpack
    nlohmann::json resp;
    mtx.lock();
    resp["time"] = nlohmann::json(times);
    resp["sinza"] = nlohmann::json(sinza);
    resp["ringmap"] = nlohmann::json(map.at(freqs[f_ind].first).at(pol));
    resp["weight"] = nlohmann::json(wgt.at(freqs[f_ind].first).at(pol));
    std::vector<std::uint8_t> resp_msgpack = nlohmann::json::to_msgpack(resp);
    mtx.unlock();
    conn.send_binary_reply(resp_msgpack.data(), resp_msgpack.size());
    return;
}

void RingMapMaker::change_dataset_state(dset_id_t new_ds_id) {

    // Update stored ID
    ds_id = new_ds_id;

    auto& dm = datasetManager::instance();

    // Get the product spec and (if available) the stackState to determine the
    // number of vis entries we are expecting
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const stackState* sstate = sstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    if (pstate == nullptr || istate == nullptr || fstate == nullptr) {
        FATAL_ERROR("Could not find all dataset states for incoming dataset with ID {}."
                    "\nOne of them is a nullptr: prod {}, input {}, freq {}.",
                    ds_id, pstate != nullptr, istate != nullptr, fstate != nullptr);
        return;
    }

    if (sstate == nullptr) {
        FATAL_ERROR("RingMapMaker requires visibilities stacked.");
    }

    stacks = sstate->get_stack_map();
    prods = pstate->get_prods();
    inputs = istate->get_inputs();
    freqs = fstate->get_freqs();

    num_stack = sstate->get_num_stack();
}

bool RingMapMaker::setup(size_t frame_id) {

    // Wait for the input buffer to be filled with data
    if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
        return false;
    }

    auto in_frame = VisFrameView(in_buf, frame_id);
    ds_id = in_frame.dataset_id;
    change_dataset_state(ds_id);

    // TODO: make these config options ?
    float fpga_s = Telescope::instance().seq_length_nsec() * 1e-9;
    num_pix = 511; // # unique NS baselines
    num_pol = 4;
    num_time = 24. * 3600. / (in_frame.fpga_seq_length * fpga_s);
    num_bl = (num_stack + 1) / 4;

    sinza = std::vector<float>(num_pix, 0.);
    for (int i = 0; i < (int)num_pix; i++) {
        sinza[i] = (i - (int)num_pix / 2) * 2. / num_pix;
    }

    // generate map making matrices
    gen_matrices();

    // initialize map containers
    mtx.lock();
    for (auto f : freqs) {
        std::vector<std::vector<float>> vis(num_pol);
        std::vector<std::vector<float>> w(num_pol);
        for (uint p = 0; p < num_pol; p++) {
            vis.at(p).resize(num_time * num_pix);
            w.at(p).resize(num_time);
            std::fill(vis.at(p).begin(), vis.at(p).end(), 0.);
            std::fill(w.at(p).begin(), w.at(p).end(), 0.);
        }
        map.insert(std::pair<uint64_t, std::vector<std::vector<float>>>(f.first, vis));
        wgt.insert(std::pair<uint64_t, std::vector<std::vector<float>>>(f.first, w));
    }
    mtx.unlock();

    // Make sure times are empty
    times.clear();
    times_map.clear();

    // Initialize the time indexing
    min_ctime = ts_to_double(std::get<1>(in_frame.time));
    max_ctime = min_ctime;
    latest = modulo<size_t>(num_time);

    return true;
}

void RingMapMaker::gen_matrices() {

    // calculate baseline for every stacked product
    ns_baselines.resize(num_bl);
    chimeFeed input_a, input_b;
    float max_bl = 0.;
    size_t auto_ind = 0;
    for (size_t i = 0; i < num_bl; i++) {
        stack_ctype s = stacks[i];
        input_ctype ia = inputs[prods[s.prod].input_a];
        input_ctype ib = inputs[prods[s.prod].input_b];
        input_a = chimeFeed::from_input(ia);
        input_b = chimeFeed::from_input(ib);
        ns_baselines[i] = feed_sep * (input_b.feed_location - input_a.feed_location);
        if (s.conjugate)
            ns_baselines[i] *= -1;
        if (std::abs(ns_baselines[i]) > max_bl)
            max_bl = std::abs(ns_baselines[i]);
        if (ia.chan_id == ib.chan_id)
            auto_ind = i;
    }

    std::vector<float> apod_coeff = apod(ns_baselines, max_bl, apodization);
    float norm = 0.;
    for (float a : apod_coeff) {
        norm += a;
    }
    if (norm == 0.)
        norm = 1.;
    if (exclude_autos)
        apod_coeff[auto_ind] = 0.;

    // Construct matrix of phase weights for every baseline and pixel
    for (auto f : freqs) {
        std::vector<cfloat> m(num_pix * num_bl);
        float lam = wl(f.second.centre);
        for (uint p = 0; p < num_pix; p++) {
            for (uint i = 0; i < num_bl; i++) {
                m[p * num_bl + i] = std::exp(cfloat(-2.i) * pi * ns_baselines[i] / lam * sinza[p])
                                    * apod_coeff[i] / norm;
            }
        }
        vis2map.insert(std::pair<uint64_t, std::vector<cfloat>>(f.first, m));
    }
}

int64_t RingMapMaker::resolve_time(time_ctype t) {

    if (t.ctime < min_ctime) {
        // time is too old, discard
        WARN("Frame older than oldest time in ringmap. Discarding.");
        return -1;
    }

    if (t.ctime > max_ctime) {
        mtx.lock();
        // We need to add a new time
        max_ctime = t.ctime;
        // Increment position
        if (times_map.size() < num_time) {
            // Still filling in the array
            if (times_map.size() > 0)
                latest++;
            times.push_back(t);
        } else {
            // Remove oldest entry
            min_ctime = times[++latest].ctime;
            times_map.erase(min_ctime);
            times[latest] = t;
        }
        times_map.insert(std::pair<double, size_t>(t.ctime, latest));
        // Clear maps
        size_t start = latest;
        size_t stop = size_t(latest) + 1;
        for (auto f : freqs) {
            uint64_t fid = f.first;
            for (uint p = 0; p < num_pol; p++) {
                std::fill(map.at(fid).at(p).begin() + start * num_pix,
                          map.at(fid).at(p).begin() + stop * num_pix, 0.);
                std::fill(wgt.at(fid).at(p).begin() + start, wgt.at(fid).at(p).begin() + stop, 0.);
            }
        }
        mtx.unlock();

        return latest;
    }

    // Otherwise find the existing time
    auto res = times_map.find(t.ctime);
    if (res == times_map.end()) {
        // No entry for this time
        WARN("Could not find this time in ringmap. Discarding.");
        return -1;
    }
    return res->second;
}

RedundantStack::RedundantStack(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RedundantStack::main_thread, this)) {

    // Get buffers from config
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void RedundantStack::change_dataset_state(dset_id_t ds_id) {
    auto& dm = datasetManager::instance();
    state_id_t stack_state_id;

    // Get input & prod states synchronoulsy
    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);
    const inputState* input_state_ptr = istate_fut.get();
    const prodState* prod_state_ptr = pstate_fut.get();
    old_stack_state_ptr = sstate_fut.get();

    if (input_state_ptr == nullptr) {
        FATAL_ERROR("Could not find inputState for incoming dataset with ID {}.", ds_id);
        return;
    }
    if (prod_state_ptr == nullptr) {
        FATAL_ERROR("Could not find prodState for incoming dataset with ID {}.", ds_id);
        return;
    }
    if (old_stack_state_ptr == nullptr) {
        FATAL_ERROR("Could not find stackState for incoming dataset with ID {}.", ds_id);
        return;
    }

    auto sspec = full_redundant(input_state_ptr->get_inputs(), prod_state_ptr->get_prods());
    auto sstate = std::make_unique<stackState>(sspec.first, std::move(sspec.second));

    std::tie(stack_state_id, new_stack_state_ptr) = dm.add_state(std::move(sstate));

    output_dset_id = dm.add_dataset(stack_state_id, ds_id);
}

void RedundantStack::main_thread() {

    frameID in_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    // Wait for the input buffer to be filled with data
    if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr) {
        return;
    }

    auto input_frame = VisFrameView(in_buf, in_frame_id);
    input_dset_id = input_frame.dataset_id;
    change_dataset_state(input_dset_id);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = VisFrameView(in_buf, in_frame_id);

        // Check dataset id hasn't changed
        if (input_frame.dataset_id != input_dset_id) {
            WARN("Input dataset ID has changed. Regenerating stack specs.");
            input_dset_id = input_frame.dataset_id;
            change_dataset_state(input_dset_id);
        }

        const auto& stack_rmap = new_stack_state_ptr->get_rstack_map();
        const auto& old_stack_map = old_stack_state_ptr->get_stack_map();
        auto num_stack = new_stack_state_ptr->get_num_stack();

        std::vector<float> stack_norm(new_stack_state_ptr->get_num_stack(), 0.0);
        std::vector<float> stack_v2(new_stack_state_ptr->get_num_stack(), 0.0);

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Create view to output frame
        auto output_frame = VisFrameView::create_frame_view(
            out_buf, output_frame_id, input_frame.num_elements, num_stack, input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {VisField::vis, VisField::weight});
        output_frame.dataset_id = output_dset_id;

        // Zero the output frame
        std::fill(std::begin(output_frame.vis), std::end(output_frame.vis), 0.0);
        std::fill(std::begin(output_frame.weight), std::end(output_frame.weight), 0.0);

        auto in_vis = input_frame.vis.data();
        auto out_vis = output_frame.vis.data();
        auto in_weight = input_frame.weight.data();
        auto out_weight = output_frame.weight.data();

        // Iterate over all the products and average together
        for (uint32_t old_ind = 0; old_ind < old_stack_map.size(); old_ind++) {
            // TODO: if the weights are ever different from 0 or 1, we will
            // definitely need to rewrite this.

            // Alias the parts of the data we are going to stack
            cfloat vis = in_vis[old_ind];
            float weight = in_weight[old_ind];

            auto& old_s = old_stack_map[old_ind];
            auto& s = stack_rmap[old_s.prod];

            // If the weight is zero, completey skip this iteration
            if (weight == 0)
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
        for (uint32_t stack_ind = 0; stack_ind < num_stack; stack_ind++) {

            // Calculate the mean and accumulate weight and place in the frame
            float norm = stack_norm[stack_ind];

            // Invert norm if set, otherwise use zero to set data to zero.
            float inorm = (norm != 0.0) ? (1.0 / norm) : 0.0;

            output_frame.vis[stack_ind] *= inorm;
            float iwgt =
                ((output_frame.weight[stack_ind] != 0.0) ? 1.0 / output_frame.weight[stack_ind]
                                                         : 0.0);
            output_frame.weight[stack_ind] = norm * norm * iwgt;

            // Accumulate to calculate the variance of the residuals
            vart += stack_v2[stack_ind] - std::norm(output_frame.vis[stack_ind]) * norm;
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
std::pair<feed_diff, bool> calculate_chime_vis_full(const prod_ctype& p,
                                                    const std::vector<input_ctype>& inputs) {

    chimeFeed fa = chimeFeed::from_input(inputs[p.input_a]);
    chimeFeed fb = chimeFeed::from_input(inputs[p.input_b]);

    bool is_wrong_cylorder = (fa.cylinder > fb.cylinder);
    bool is_same_cyl_wrong_feed_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location > fb.feed_location));
    bool is_same_feed_wrong_pol_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location == fb.feed_location)
         && (fa.polarisation > fb.polarisation));

    bool conjugate = false;

    // Check if we need to conjugate/transpose to get the correct order
    if (is_wrong_cylorder || is_same_cyl_wrong_feed_order || is_same_feed_wrong_pol_order) {

        chimeFeed t = fa;
        fa = fb;
        fb = t;
        conjugate = true;
    }

    return {std::make_tuple(fa.polarisation, fb.polarisation, fb.cylinder - fa.cylinder,
                            fb.feed_location - fa.feed_location),
            conjugate};
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
