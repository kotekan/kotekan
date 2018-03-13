#include "prodSubset.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"

REGISTER_KOTEKAN_PROCESS(prodSubset);

prodSubset::prodSubset(Config &config,
                               const string& unique_name,
                               bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&prodSubset::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int(unique_name, "num_elements");
    num_eigenvectors =  config.get_int(unique_name, "num_ev");

    size_t num_prod = num_elements * (num_elements + 1) / 2;

    // Get buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    // TODO: Size of buffer is not adjusted for baseline subset.
    //       ~ 3/4 of buffer will be unused.
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    prod_ind = std::get<0>(parse_prod_subset(config, unique_name));

    subset_num_prod = prod_ind.size();
}

void prodSubset::apply_config(uint64_t fpga_seq) {

}

void prodSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

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
                                     num_elements, subset_num_prod,
                                     num_eigenvectors);

        // Copy over subset of visibilities
        for (size_t i = 0; i < subset_num_prod; i++) {
            output_frame.vis[i] = input_frame.vis[prod_ind[i]];
            output_frame.weight[i] = input_frame.weight[prod_ind[i]];
        }

        // Copy the non-visibility parts of the buffer
        output_frame.copy_nonvis_buffer(input_frame);
        // Copy metadata
        output_frame.copy_nonconst_metadata(input_frame);

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame id
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;

    }
}

inline bool max_bl_condition(prod_ctype prod, int xmax, int ymax) {

    // Figure out feed separations
    int x_sep = prod.input_a / 512 - prod.input_b / 512;
    int y_sep = prod.input_a % 256 - prod.input_b % 256;
    if (x_sep < 0) x_sep = - x_sep;
    if (y_sep < 0) y_sep = - y_sep;

    return (x_sep <= xmax) && (y_sep <= ymax);
}

inline bool max_bl_condition(uint32_t vis_ind, int n, int xmax, int ymax) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return max_bl_condition(prod, xmax, ymax);
}

inline bool have_inputs_condition(prod_ctype prod, 
                                std::vector<int> input_list) {
   
    bool prod_in_list = false;
    for(auto ipt : input_list) {
        if ((prod.input_a==ipt) || (prod.input_b==ipt)) {
            prod_in_list = true;
            break;
        }
    }

    return prod_in_list;
}

inline bool have_inputs_condition(uint32_t vis_ind, int n, 
                                std::vector<int> input_list) {
    
    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return have_inputs_condition(prod, input_list);
}

inline bool only_inputs_condition(prod_ctype prod, 
                                std::vector<int> input_list) {
   
    bool ipta_in_list = false;
    bool iptb_in_list = false;
    for(auto ipt : input_list) {
        if (prod.input_a==ipt) {
            ipta_in_list = true;
        }
        if (prod.input_b==ipt) {
            iptb_in_list = true;
        }
    }

    return (ipta_in_list && iptb_in_list);
}

inline bool only_inputs_condition(uint32_t vis_ind, int n, 
                                std::vector<int> input_list) {
    
    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return only_inputs_condition(prod, input_list);
}


std::tuple<std::vector<size_t>, std::vector<prod_ctype>>
parse_prod_subset(Config& config, const std::string base_path) {

    size_t num_elements = config.get_int(base_path, "num_elements");
    std::vector<size_t> prod_ind_vec;
    std::vector<prod_ctype> prod_ctype_vec;

    // Type of product selection based on config parameter
    std::string prod_subset_type = config.get_string_default(base_path, "prod_subset_type", "all");


    if (prod_subset_type == "autos") {
        for (int ii=0; ii<num_elements; ii++) {
            prod_ind_vec.push_back(cmap(ii,ii,num_elements));
            prod_ctype_vec.push_back({ii,ii});
//            prod_ctype_vec.emplace_back((prod_ctype){ii,ii});
        }
    } else if (prod_subset_type == "baseline") {
        // Define criteria for baseline selection based on config parameters
        uint16_t xmax, ymax;
        xmax = config.get_int(base_path, "max_ew_baseline");
        ymax = config.get_int(base_path, "max_ns_baseline");
        // Find the products in the subset
        for (int ii=0; ii<num_elements; ii++) {
            for (int jj=ii; jj<num_elements; jj++) {
                if (max_bl_condition((prod_ctype){ii,jj}, xmax, ymax)) {
                    prod_ind_vec.push_back(cmap(ii,jj,num_elements));
                    prod_ctype_vec.push_back({ii,jj});
//                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "have_inputs") {
        std::vector<int> input_list;
        input_list = config.get_int_array(base_path, "input_list");
        // Find the products in the subset
        for (int ii=0; ii<num_elements; ii++) {
            for (int jj=ii; jj<num_elements; jj++) {
                if (have_inputs_condition((prod_ctype){ii,jj}, input_list)) {
                    prod_ind_vec.push_back(cmap(ii,jj,num_elements));
                    prod_ctype_vec.push_back({ii,jj});
//                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "only_inputs") {
        std::vector<int> input_list;
        input_list = config.get_int_array(base_path, "input_list");
        // Find the products in the subset
        for (int ii=0; ii<num_elements; ii++) {
            for (int jj=ii; jj<num_elements; jj++) {
                if (only_inputs_condition((prod_ctype){ii,jj}, input_list)) {
                    prod_ind_vec.push_back(cmap(ii,jj,num_elements));
                    prod_ctype_vec.push_back({ii,jj});
//                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "all") {
        // Find the products in the subset
        for (int ii=0; ii<num_elements; ii++) {
            for (int jj=ii; jj<num_elements; jj++) {
                prod_ind_vec.push_back(cmap(ii,jj,num_elements));
                prod_ctype_vec.push_back({ii,jj});
//              prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
            }
        }
    }

    return std::make_tuple(prod_ind_vec, prod_ctype_vec);

}
