#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <vector>

#include "json.hpp"

// Name space includes.
using json = nlohmann::json;
using std::string;
using std::vector;

class Config {
public:
    Config();
    Config(const Config& orig);
    virtual ~Config();

    // ----------------------------
    // Get config value functions.
    // ----------------------------

    int32_t get_int(const string &pointer);
    float get_float(const string &pointer);
    double get_double(const string &pointer);
    string get_string(const string &pointer);
    bool get_bool(const string &pointer);
    vector<int32_t> get_int_array(const string &pointer);
    vector<float> get_float_array(const string &pointer);
    vector<double> get_double_array(const string &pointer);
    vector<string> get_string_array(const string &pointer);
    vector<json> get_json_array(const string &pointer);

    void parse_file(const string &file_name, uint64_t switch_fpga_seq);

    // @param updates Json object with values to be replaced.
    // @param start_fpga_seq The fpga seq number to update the config on.
    // This value must be in the future.
    void update_config(json updates, uint64_t start_fpga_seq);

    // Returns true if that fpga_seq number matches the switch_fpga_seq value.
    // i.e. you need to reload the values for this config.
    bool update_needed(uint32_t fpga_seq);

    uint64_t get_switch_fpga_seq();

    // This function should be moved, it doesn't really belong here...
    int32_t num_links_per_gpu(const int32_t &gpu_id);

    // This is an odd function that existed in the old config,
    // it should be moved out of this object at some point.
    void generate_extra_options();

    // Debug
    void dump_config();
private:

    json _json[2];
    int32_t _gain_bank;
    // Switch the once this fpga sequence number is reached.
    uint64_t _switch_fpga_seq = 0;
};

#endif /* CONFIG_HPP */

