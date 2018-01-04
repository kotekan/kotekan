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

    int32_t get_int(const string& base_path, const string& name);
    // Same as get_int, but if it cannot find the value, it returns `default_value`
    int32_t get_int_default(const string& base_path, const string& name, int32_t default_value);
    // Treats the value as an arithmetic expression with other variables,
    // and return the result of evaluating it.
    int32_t get_int_eval(const string& base_path, const string& name);

    uint64_t get_uint64(const string& base_path, const string& name);

    float get_float(const string& base_path, const string& name);
    // Same as get_float, but if it cannot find the value, it returns `default_value`
    float get_float_default(const string& base_path, const string& name, float default_value);

    double get_double(const string& base_path, const string& name);
    // Same as get_double, but if it cannot find the value, it returns `default_value`
    double get_double_default(const string& base_path, const string& name, double default_value);
    // Treats the value as an arithmetic expression with other variables,
    // and return the result of evaluating it.
    double get_double_eval(const string& base_path, const string& name);

    string get_string(const string& base_path, const string& name);
    // Same as get_string, but if it cannot find the value, it returns `default_value`
    string get_string_default(const string& base_path, const string& name, const string& default_value);

    bool get_bool(const string& base_path, const string& name);
    // Same as get_bool, but if it cannot find the value, it returns `default_value`
    bool get_bool_default(const string& base_path, const string& name, bool default_value);

    // Returns true if the path exists
    bool exists(const string& base_path, const string& name);

    vector<int32_t> get_int_array(const string& base_path, const string& name);
    vector<float> get_float_array(const string& base_path, const string& name);
    vector<double> get_double_array(const string& base_path, const string& name);
    vector<string> get_string_array(const string& base_path, const string& name);
    vector<json> get_json_array(const string& base_path, const string& name);

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

    // @breaf Finds the value with key "name" starts looking at the
    // "base_pointer" location, and then works backwards up the config tree.
    // @param base_pointer Contains a JSON pointer which points to the
    // process's location in the config tree. i.e. /vdif_cap/disk_write
    // @param name The name of the property i.e. num_frequencies
    json get_value(const string &base_pointer, const string &name);

    // Returns the full json data structure (for internal framework use)
    json &get_full_config_json();

    // Debug
    void dump_config();
private:

    json _json[2];
    int32_t _gain_bank;
    // Switch the once this fpga sequence number is reached.
    uint64_t _switch_fpga_seq = 0;
};

#endif /* CONFIG_HPP */

