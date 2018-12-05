
#ifndef RINGMAP_HPP
#define RINGMAP_HPP

#include "gsl-lite.hpp"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "restServer.hpp"

class mapMaker : public KotekanProcess {

public:

    // Default constructor
    mapMaker(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq) override;

    // Main loop for the process
    void main_thread() override;

    /// REST endpoint to request a map
    nlohmann::json rest_callback(connectionInstance& conn, nlohmann::json &json);

    /// Abbreviation for RingMap type
    typedef RingMap std::vector<std::vector<cfloat>>;

private:

    bool setup();

    void gen_matrices();

    void gen_baselines();

    size_t append_time(time_ctype t);

    inline float wl(uint32_t fid);

    // Matrix from visibilities to map for every freq (same for each pol)
    std::map<uint32_t,std::vector<cfloat>> vis2map;
    // Store the maps and weight maps for every frequency
    std::map<uint32_t,RingMap> map;
    std::map<uint32_t,RingMap> wgt_map;
    std::vector<float> baselines;

    // Keep track of map dimensions
    std::vector<float> sinza;
    std::vector<time_ctype> times;
    std::map<uint64_t, size_t> times_map;
    uint64_t max_fpga, min_fpga;

    uint32_t num_time;
    uint32_t num_pix;
    uint8_t num_pol;
    uint32_t num_stack;
    uint32_t num_bl;

    std::vector<uint32_t> excl_input;
    std::vector<uint32_t> freq_id;
    std::vector<prod_ctype> prod;

    // Buffer to read from
    Buffer* in_buf;
};

#endif