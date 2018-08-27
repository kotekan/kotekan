#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <iostream>

#include "visUtil.hpp"
#include "visBuffer.hpp"
#include "visCompression.hpp"
#include "fmt.hpp"

using namespace std::placeholders;

REGISTER_KOTEKAN_PROCESS(baselineCompression);
REGISTER_DATASET_STATE(stackState);

baselineCompression::baselineCompression(Config &config,
                                         const string& unique_name,
                                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&baselineCompression::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");

    // Fill out the map of stack types
    stack_type_defs["diagonal"] = stack_diagonal;
    stack_type_defs["chime_in_cyl"] = stack_chime_in_cyl;

    std::string stack_type = config.get_string(unique_name, "stack_type");
    if(stack_type_defs.count(stack_type) == 0) {
        ERROR("unknown stack type %s", stack_type.c_str());
        return;
    }
    INFO("using stack type: %s", stack_type.c_str());
    calculate_stack = stack_type_defs.at(stack_type);

}


void baselineCompression::apply_config(uint64_t fpga_seq) {

}


void baselineCompression::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    // TODO: put in a real map here
    // Create a product map (assuming full N^2) eventually this will be fetched
    // using the dataset manager.
    std::vector<prod_ctype> prods;
    for(uint16_t i = 0; i < num_elements; i++) {
        for(uint16_t j = i; j < num_elements; j++) {
            prods.push_back({i, j});
        }
    }
    std::vector<input_ctype> inputs(num_elements);

    /// Map of product index in the input stream to (output index, conjugate)
    std::vector<rstack_ctype> stack_map;

    /// The number of stacks in the output
    uint32_t num_stack;

    // Calculate the stack definition
    std::tie(num_stack, stack_map) = calculate_stack(inputs, prods);

    // Keep track of the normalisation of each stack
    std::vector<float> stack_norm;

    auto& dm = datasetManager::instance();
    const stackState * stack_state_ptr;
    dset_id input_dset_id = -1;
    dset_id output_dset_id;
    state_id stack_state_id;


    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer frame to be free
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                output_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // If the input dataset has changed construct a new stack spec from the
        // datasetManager
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;

            auto istate = dm.closest_ancestor_of_type<inputState>(input_dset_id);
            auto pstate = dm.closest_ancestor_of_type<prodState>(input_dset_id);

            auto sspec = calculate_stack(istate.second->get_inputs(),
                                         pstate.second->get_prods());
            auto sstate = std::make_unique<stackState>(
                sspec.first, std::move(sspec.second));

            std::tie(stack_state_id, stack_state_ptr) =
                dm.add_state(std::move(sstate));
            output_dset_id = dm.add_dataset(stack_state_id, input_dset_id);

            stack_norm = std::vector<float>(stack_state_ptr->get_num_stack());
        }

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);
        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         input_frame.num_elements,
                                         stack_state_ptr->get_num_stack(),
                                         input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_nonconst_metadata(input_frame);
        output_frame.copy_nonvis_buffer(input_frame);

        // Reset the normalisation array and zero the output frame
        std::fill(std::begin(stack_norm), std::end(stack_norm), 0);
        std::fill(std::begin(output_frame.vis), std::end(output_frame.vis), 0.0);
        std::fill(std::begin(output_frame.weight),
                  std::end(output_frame.weight), 0.0);

        auto stack_map = stack_state_ptr->get_rstack_map();

        // Iterate over all the products and average together
        for(uint32_t prod_ind = 0; prod_ind < prods.size(); prod_ind++) {

            auto& p = prods[prod_ind];
            auto& s = stack_map[prod_ind];

            // Alias the parts of the data we are going to stack
            float weight = input_frame.weight[prod_ind];
            cfloat vis = input_frame.vis[prod_ind];
            vis = s.conjugate ? conj(vis) : vis;

            // Set the weighting used to combine baselines
            float w = (weight != 0) *
                input_frame.flags[p.input_a] * input_frame.flags[p.input_b];

            // First summation of the visibilities (dividing by the total weight will be done later)
            output_frame.vis[s.stack] += w * vis;

            // Accumulate the weighted *variances*. Normalising and inversion
            // will be done later
            // NOTE: hopefully there aren't too many zeros so the branch
            // predictor will work well
            output_frame.weight[s.stack] += (w == 0) ? 0 : (w * w / weight);

            // Accumulate the weights so we can normalize correctly
            stack_norm[s.stack] += w;
        }

        // Loop over the stacks and normalise (and invert the variances)
        for(uint32_t stack_ind = 0; stack_ind < num_stack; stack_ind++) {
            float norm = stack_norm[stack_ind];
            output_frame.vis[stack_ind] /= norm;
            output_frame.weight[stack_ind] = norm * norm /
                output_frame.weight[stack_ind];
        }

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame id
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}

// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>> stack_diagonal(
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods
) {
    uint32_t num_elements = inputs.size();
    std::vector<rstack_ctype> stack_def;

    for(auto& p : prods) {
        uint32_t stack_ind = abs(p.input_b - p.input_a);
        bool conjugate = p.input_a > p.input_b;

        stack_def.push_back({stack_ind, conjugate});
    }

    return {num_elements, stack_def};

}


chimeFeed chimeFeed::from_input(input_ctype input) {
    chimeFeed feed;

    if(input.chan_id >= 2048) {
        throw std::invalid_argument("Channel ID is not a valid CHIME feed.");
    }

    feed.cylinder = (input.chan_id / 512);
    feed.polarisation = ((input.chan_id / 256 + 1) % 2);
    feed.feed_location = input.chan_id % 256;

    return feed;
}


std::ostream & operator<<(std::ostream &os, const chimeFeed& f) {
    char cyl_name[4] = {'A', 'B', 'C', 'D'};
    char pol_name[2] = {'X', 'Y'};
    return os << fmt::format("{}{:03}{}", cyl_name[f.cylinder],
                             f.feed_location, pol_name[f.polarisation]);
}

using feed_diff = std::tuple<int8_t, int8_t, int8_t, int8_t, int16_t>;

// Calculate the baseline parameters and whether the product must be
// conjugated to get canonical ordering
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
        {fa.polarisation, fb.polarisation, fa.cylinder, fb.cylinder,
         fb.feed_location - fa.feed_location},
         conjugate
    };
}

// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>> stack_chime_in_cyl(
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods
) {
    // Calculate the set of baseline properties
    std::vector<std::pair<feed_diff, bool>> bl_prop;
    std::transform(std::begin(prods), std::end(prods),
                   std::back_inserter(bl_prop),
                   std::bind(calculate_chime_vis, _1, inputs));

    // Create an index array for doing the sorting
    std::vector<uint32_t> sort_ind(prods.size());
    std::iota(std::begin(sort_ind), std::end(sort_ind), 0);

    auto sort_fn = [&](const uint32_t& ii, const uint32_t& jj) -> bool {
        return (bl_prop[ii].first <
                bl_prop[jj].first);
    };
    std::sort(std::begin(sort_ind), std::end(sort_ind), sort_fn);

    std::vector<rstack_ctype> stack_map(prods.size());

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


std::vector<stack_ctype> invert_stack(
    uint32_t num_stack, const std::vector<rstack_ctype>& stack_map)
{
    std::vector<stack_ctype> res(num_stack);
    size_t num_prod = stack_map.size();

    for(uint32_t i = 0; i < num_prod; i++) {
        uint32_t j = num_prod - i - 1;
        res[stack_map[j].stack] = {j, stack_map[j].conjugate};
    }

    return res;
}