#include <cstdlib>
#include <vector>
#include <algorithm>

#include "visUtil.hpp"
#include "visBuffer.hpp"
#include "visCompression.hpp"


REGISTER_KOTEKAN_PROCESS(baselineCompression);


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

    std::string stack_type = config.get_string(unique_name, "stack_type");
    INFO("using stack type: %s", stack_type.c_str());
    if(stack_type_defs.count(stack_type) == 0) {
        ERROR("unknown stack type %s", stack_type.c_str());
    }
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
    std::vector<std::pair<uint32_t, bool>> stack_map;

    /// The number of stacks in the output
    uint32_t num_stack;

    // Calculate the stack definition
    std::tie(num_stack, stack_map) = calculate_stack(inputs, prods);

    // Keep track of the normalisation of each stack
    std::vector<float> stack_norm(num_stack);

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

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);
        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         input_frame.num_elements, num_stack,
                                         input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_nonconst_metadata(input_frame);
        output_frame.copy_nonvis_buffer(input_frame);

        // Reset the normalisation array and zero the output frame
        std::fill(stack_norm.begin(), stack_norm.end(), 0);
        std::fill(output_frame.vis.begin(), output_frame.vis.end(), 0.0);
        std::fill(output_frame.weight.begin(), output_frame.weight.end(), 0.0);

        // Iterate over all the products and average together
        for(uint32_t prod_ind = 0; prod_ind < prods.size(); prod_ind++) {

            uint32_t stack_ind;
            bool conjugate;

            auto& p = prods[prod_ind];
            std::tie(stack_ind, conjugate) = stack_map[prod_ind];

            // Alias the parts of the data we are going to stack
            float weight = input_frame.weight[prod_ind];
            cfloat vis = input_frame.vis[prod_ind];
            vis = conjugate ? conj(vis) : vis;

            // Set the weighting used to combine baselines
            float w = (weight != 0) *
                input_frame.flags[p.input_a] * input_frame.flags[p.input_b];

            // First summation of the visibilities (dividing by the total weight will be done later)
            output_frame.vis[stack_ind] += w * vis;

            // Accumulate the weighted *variances*. Normalising and inversion
            // will be done later
            // NOTE: hopefully there aren't too many zeros so the branch
            // predictor will work well
            output_frame.weight[stack_ind] += (w == 0) ? 0 : (w * w / weight);

            // Accumulate the weights so we can normalize correctly
            stack_norm[stack_ind] += w;
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
std::pair<uint32_t, std::vector<std::pair<uint32_t, bool>>> stack_diagonal(
    std::vector<input_ctype>& inputs, std::vector<prod_ctype>& prods
) {
    uint32_t num_elements = inputs.size();
    std::vector<std::pair<uint32_t, bool>> stack_def;

    for(auto& p : prods) {
        uint32_t stack_ind = abs(p.input_b - p.input_a);
        bool conjugate = p.input_a > p.input_b;

        stack_def.emplace_back(stack_ind, conjugate);
    }

    INFO("Here2 %i", num_elements);

    return {num_elements, stack_def};

}