
#ifndef RINGMAP_HPP
#define RINGMAP_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "visUtil.hpp"
#include "restServer.hpp"
#include "datasetManager.hpp"
#include "fpga_header_functions.h"
#include "gsl-lite.hpp"

class mapMaker : public kotekan::Stage {

public:

    // Default constructor
    mapMaker(kotekan::Config &config,
             const string& unique_name,
             kotekan::bufferContainer &buffer_container);

    // Main loop for the process
    void main_thread() override;

    /// REST endpoint to request a map
    void rest_callback(kotekan::connectionInstance& conn,
                       nlohmann::json &json);
    void rest_callback_get(kotekan::connectionInstance& conn);

    /// Abbreviation for RingMap type
    typedef std::vector<std::vector<cfloat>> RingMap;

private:

    bool setup(size_t frame_id);

    void gen_matrices();

    int64_t resolve_time(time_ctype t);

    inline float wl(float freq) {
        return 299.792458 / freq;
    };

    // Matrix from visibilities to map for every freq (same for each pol)
    std::map<uint32_t,std::vector<cfloat>> vis2map;
    // Store the maps and weight maps for every frequency
    std::map<uint32_t,std::vector<std::vector<cfloat>>> map;
    std::map<uint32_t,std::vector<std::vector<cfloat>>> wgt_map;
    std::vector<float> ns_baselines;
    std::vector<stack_ctype> stacks;
    std::vector<prod_ctype> prods;
    std::vector<input_ctype> inputs;
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;

    // Keep track of map dimensions
    std::vector<float> sinza;
    std::vector<time_ctype> times;
    std::map<uint64_t, size_t> times_map;
    modulo<size_t> latest;
    uint64_t max_fpga, min_fpga;

    uint32_t num_time;
    uint32_t num_pix;
    uint8_t num_pol;
    uint32_t num_stack;
    uint32_t num_bl;

    dset_id_t ds_id;
    std::vector<uint32_t> excl_input;

    // Mutex for reading and writing to maps
    std::mutex mtx;

    // Map buffer file
    void * map_file;

    // Buffer to read from
    Buffer* in_buf;
};

class redundantStack : public kotekan::Stage {

public:

    redundantStack(kotekan::Config &config,
                   const string& unique_name,
                   kotekan::bufferContainer &buffer_container);

    void main_thread();

private:

    void change_dataset_state(dset_id_t ds_id);

    // dataset states and IDs
    dset_id_t output_dset_id;
    dset_id_t input_dset_id;
    const prodState* prod_state_ptr;
    const stackState* old_stack_state_ptr;
    const stackState* new_stack_state_ptr;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

std::pair<uint32_t, std::vector<rstack_ctype>> full_redundant(
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods
);

#endif
