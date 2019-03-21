
#ifndef RINGMAP_HPP
#define RINGMAP_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "datasetManager.hpp"
#include "fpga_header_functions.h"
#include "restServer.hpp"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

class mapMaker : public kotekan::Stage {

public:
    // Default constructor
    mapMaker(kotekan::Config& config, const string& unique_name,
             kotekan::bufferContainer& buffer_container);

    // Main loop for the process
    void main_thread() override;

    /// REST endpoint to request a map
    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json);
    void rest_callback_get(kotekan::connectionInstance& conn);

    /// Abbreviation for RingMap type
    typedef std::vector<std::vector<cfloat>> RingMap;

private:
    void change_dataset_state(dset_id_t ds_id);

    bool setup(size_t frame_id);

    void gen_matrices();

    int64_t resolve_time(time_ctype t);

    inline float wl(float freq) {
        return 299.792458 / freq;
    };

    // Matrix from visibilities to map for every freq (same for each pol)
    std::map<uint32_t, std::vector<cfloat>> vis2map;
    std::map<uint32_t, std::vector<float>> wgt2map;
    // Store the maps and weight maps for every frequency
    std::map<uint32_t, std::vector<std::vector<float>>> map;
    std::map<uint32_t, std::vector<std::vector<float>>> wgt;

    // Visibilities specs
    std::vector<stack_ctype> stacks;
    std::vector<prod_ctype> prods;
    std::vector<input_ctype> inputs;
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::vector<float> ns_baselines;

    // Dimensions
    uint32_t num_time;
    uint32_t num_pix;
    uint8_t num_pol;
    uint32_t num_stack;
    uint32_t num_bl;

    // Map dimensions and time keeping
    std::vector<float> sinza;
    std::vector<time_ctype> times;
    std::map<uint64_t, size_t> times_map;
    modulo<size_t> latest;
    uint64_t max_fpga, min_fpga;

    // Dataset ID of incoming stream
    dset_id_t ds_id;

    // Configurable
    float feed_sep;

    // Mutex for reading and writing to maps
    std::mutex mtx;

    // Buffer to read from
    Buffer* in_buf;
};

class redundantStack : public kotekan::Stage {

public:
    redundantStack(kotekan::Config& config, const string& unique_name,
                   kotekan::bufferContainer& buffer_container);

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

std::pair<uint32_t, std::vector<rstack_ctype>>
full_redundant(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods);

#endif
