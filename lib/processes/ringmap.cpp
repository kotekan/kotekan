#include "ringmap.hpp"
#include "visBuffer.hpp"
#include "datasetManager.hpp"
#include "visCompression.hpp"
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

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, in_frame_id);
        uint32_t f_id = input_frame.freq_id;

        // Check dataset id hasn't changed
        if (input_frame.dataset_id != ds_id) {

            // TODO: what should happen in this case?

            //string msg = fmt::format(
            //    "Unexpected dataset ID={} received (expected id={}).",
            //    input_frame.dataset_id, ds_id
            //);
            //ERROR(msg.c_str());
            //raise(SIGINT);
            //return;
        }

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

    auto frame = visFrameView(in_buf, frame_id);
    ds_id = frame.dataset_id;
    auto& dm = datasetManager::instance();

    change_dataset_state();

    // Get the product spec and (if available) the stackState to determine the
    // number of vis entries we are expecting
    auto istate = dm.dataset_state<inputState>(ds_id);
    auto pstate = dm.dataset_state<prodState>(ds_id);
    auto sstate = dm.dataset_state<stackState>(ds_id);
    auto mstate = dm.dataset_state<metadataState>(ds_id);
    if (pstate == nullptr || sstate == nullptr || mstate == nullptr || istate == nullptr)
        throw std::runtime_error("Could not find all dataset states for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".\n" \
                                 "One of them is a nullptr (0): prod "
                                 + std::to_string(pstate != nullptr)
                                 + ", stack "
                                 + std::to_string(sstate != nullptr)
                                 + ", metadata "
                                 + std::to_string(mstate != nullptr));

    // TODO: make these config options ?
    num_pix = 512; // # unique NS baselines
    num_pol = 4;
    num_time = 24. * 360. / (frame.fpga_seq_length * 2.56e-6);
    num_stack = sstate->get_num_stack();
    num_bl = num_stack / 4;

    sinza = std::vector<float>(num_pix, 0.);
    for (uint i = 0; i < num_pix; i++) {
        sinza[i] = i * 2. / num_pix - 1. + 1. / num_pix;
    }

    stacks = sstate->get_stack_map();
    prods = pstate->get_prods();
    inputs = istate->get_inputs();

    // generate map making matrices
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

    if (!sstate->is_stacked())
        throw std::runtime_error("MapMaker requires visibilities stacked ");
    num_stack = sstate->get_num_stack();
}

void mapMaker::gen_matrices() {

    // calculate baseline for every stacked product
    ns_baselines.reserve(num_bl);
    chimeFeed input_a, input_b;
    for (size_t i = 0; i < num_bl; i++) {
        stack_ctype s = stacks[i];
        input_a = chimeFeed::from_input(inputs[prods[s.prod].input_a]);
        input_b = chimeFeed::from_input(inputs[prods[s.prod].input_b]);
        ns_baselines[i] = input_b.feed_location - input_a.feed_location;
        if (s.conjugate)
            ns_baselines[i] *= -1;
    }

    // Construct matrix of phase weights for every baseline and pixel
    for (auto fid : freq_id) {
        std::vector<cfloat> m = vis2map[fid];
        m.reserve(num_pix * num_bl);
        float lam = wl(fid);
        for (uint p = 0; p < num_pix; p++) {
            for (uint i = 0; i < num_bl; i++) {
                m[p*num_bl + i] = std::exp(cfloat(-2.i) * pi * ns_baselines[i] / lam * sinza[p]);
            }
        }
    }
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

redundantStack::redundantStack(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&mapMaker::main_thread, this)) {

    // Get buffers from config
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

}

void redundantStack::change_dataset_state(dset_id_t ds_id) {
    auto& dm = datasetManager::instance();
    state_id_t stack_state_id;

    // TODO: get both states synchronoulsy?
    auto input_state_ptr = dm.dataset_state<inputState>(ds_id);
    if (input_state_ptr == nullptr)
        throw std::runtime_error("Could not find inputState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");
    prod_state_ptr = dm.dataset_state<prodState>(ds_id);
    if (prod_state_ptr == nullptr)
        throw std::runtime_error("Could not find prodState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");
    old_stack_state_ptr = dm.dataset_state<stackState>(ds_id);
    if (old_stack_state_ptr == nullptr)
        throw std::runtime_error("Could not find stackState for " \
                                 "incoming dataset with ID "
                                 + std::to_string(ds_id) + ".");

    auto sspec = calculate_stack(input_state_ptr->get_inputs(),
                                 old_stack_state_ptr->get_stack_map());
    auto sstate = std::make_unique<stackState>(
        sspec.first, std::move(sspec.second));

    std::tie(stack_state_id, new_stack_state_ptr) =
        dm.add_state(std::move(sstate));

    output_dset_id = dm.add_dataset(dataset(stack_state_id, ds_id));
}

void redundantStack::main_thread() {

    frameID in_frame_id(in_buf);

    if (!setup(in_frame_id))
        return;
    auto input_frame = visFrameView(in_buf, in_frame_id);
    input_dset_id = input_frame.dataset_id;
    try {
        change_dataset_state(input_dset_id);
    } catch (std::runtime_error& e) {
        retry_broker = true;
        WARN("redundantStack: Failure in " \
             "datasetManager, retrying: %s", e.what());
    }

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
            // TODO: what to do then ?
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
        output_frame.copy_nonconst_metadata(input_frame);
        output_frame.copy_nonvis_buffer(input_frame);
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

            auto& s = new_stack_rmap[old_ind];
            auto& old_s = old_stack_map[old_ind];
            auto& p = prods[old_s.prod];

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
            output_frame.weight[stack_ind] = norm * norm /
                output_frame.weight[stack_ind];

            // Accumulate to calculate the variance of the residuals
            vart += stack_v2[stack_ind]
                    - std::norm(output_frame.vis[stack_ind]) * norm;
            normt += norm;
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

}

using feed_diff = std::tuple<int8_t, int8_t, int8_t, int16_t>;

// Modified to operate on a previous stack
std::pair<uint32_t, std::vector<rstack_ctype>> calculate_stack(
    const std::vector<input_ctype>& inputs,
    const std::vector<stack_ctype>& old_stacks
) {
    // Calculate the set of baseline properties
    std::vector<std::pair<feed_diff, bool>> bl_prop;
    std::transform(std::begin(old_stacks), std::end(old_stacks),
                   std::back_inserter(bl_prop),
                   std::bind(calculate_chime_vis, _1.prod, inputs));

    // Create an index array for doing the sorting
    std::vector<uint32_t> sort_ind(old_stacks.size());
    std::iota(std::begin(sort_ind), std::end(sort_ind), 0);

    auto sort_fn = [&](const uint32_t& ii, const uint32_t& jj) -> bool {
        return (bl_prop[ii].first <
                bl_prop[jj].first);
    };
    std::sort(std::begin(sort_ind), std::end(sort_ind), sort_fn);

    std::vector<rstack_ctype> stack_map(old_stacks.size());

    feed_diff cur = bl_prop[sort_ind[0]].first;
    uint32_t cur_stack_ind = 0;

    for(auto& ind : sort_ind) {
        if(bl_prop[ind].first != cur) {
            cur = bl_prop[ind].first;
            cur_stack_ind++;
        }
        stack_map[ind] = {cur_stack_ind, bl_prop[ind].second};
    }

    return {++cur_stack_ind, stack_map};
}

// Calculate the baseline parameters and whether the product must be
// conjugated to get canonical ordering
// Modified to return cylinder separation
std::pair<feed_diff, bool> calculate_chime_vis(
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
