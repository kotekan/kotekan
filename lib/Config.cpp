#include "Config.hpp"
#include "errors.h"

#include <iostream>
#include <fstream>
#include <json.hpp>
#include <stdexcept>
#include <vector>

using std::vector;

Config::Config() {
}

Config::Config(const Config& orig) {
}

Config::~Config() {
    _json[0].clear();
}

void Config::parse_file(const string& file_name, uint64_t switch_fpga_seq) {
    try {
        std::ifstream config_file_stream(file_name);
        config_file_stream >> _json[0];
    } catch (std::exception const & ex) {
        WARN("Could not parse json file: %s, error: %s", file_name.c_str(), ex.what());
        throw ex;
    }
    _switch_fpga_seq = switch_fpga_seq;
}

int32_t Config::get_int(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_number_integer() && !value.is_number_float() ) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an integer or doesn't exist");
    }
    return value.get<int32_t>();
}

double Config::get_double(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_number_float()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a double or doesn't exist");
    }
    return value.get<double>();
}

float Config::get_float(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_number_float()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a float or doesn't exist");
    }
    return value.get<float>();
}

string Config::get_string(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_string()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a string or doesn't exist");
    }
    return value.get<string>();
}

bool Config::get_bool(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_boolean()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a boolean or doesn't exist");
    }
    return value.get<bool>();
}

vector<int32_t> Config::get_int_array(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }
    return value.get< vector<int32_t> >();
}

vector<double> Config::get_double_array(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }
    return value.get< vector<double> >();
}

vector<float> Config::get_float_array(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }
    return value.get< vector<float> >();
}

vector<string> Config::get_string_array(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }
    return value.get< vector<string> >();
}

vector<json> Config::get_json_array(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }

    return value.get< vector<json> >();
}

uint64_t Config::get_switch_fpga_seq() {
    return _switch_fpga_seq;
}

void Config::update_config(json updates, uint64_t switch_fpga_seq) {

    // TODO enable config banks
    _json[0] = updates;
    (void)switch_fpga_seq;
    // Switch gain banks here.
    return;
}

bool Config::update_needed(uint32_t fpga_seq) {
    if (fpga_seq == 0)
        return true;
    else
        return false;
}

int32_t Config::num_links_per_gpu(const int32_t& gpu_id) {

    int32_t num_links = get_int("/dpdk/", "num_links");
    vector<int32_t> link_map = get_int_array("/gpu", "link_map");
    int32_t gpus_in_link = 0;

    for (int i = 0; i < num_links; ++i) {
        if (link_map[i] == gpu_id)
            gpus_in_link++;
    }
    return gpus_in_link;
}

json Config::get_value(const string& base_path, const string& name) {
    string search_path = base_path;
    // I could make this a fancy recursive function, but this works just as well.
    for (;;) {
        json::json_pointer search_pointer(search_path);
        // Check if the search_path exists.
        try {
            // Yes this statement really does something.
            _json[0][search_pointer];
        } catch (std::exception const & ex) {
            throw std::runtime_error("The base path " + base_path + " does not exist in the config.");
        }
        // Check if the value we want exists in the search path.
        if (_json[0][search_pointer].count(name)) {
            return _json[0][search_pointer][name];
        }

        if (search_path == "")
            break;

        std::size_t last_slash = search_path.find_last_of("/");
        search_path = search_path.substr(0, last_slash);
    }
    throw std::runtime_error("The value " + name + " does not exist in the path to: " + base_path);
}

void Config::generate_extra_options() {

    // Create the inverse remap
    vector<int32_t> remap = get_int_array("/processing", "product_remap");
    vector<int32_t> inverse_remap;
    inverse_remap.resize(remap.size());

    // Given a channel ID, where is it in FPGA order.
    for(uint32_t i = 0; i < remap.size(); ++i) {
        inverse_remap[remap[i]] = i;
    }
    _json[0]["processing"]["inverse_product_remap"] = inverse_remap;

    // Special case for 16-element version
//    if (_json[0]["processing"]["num_elements"].get<int>() < 32) {
//        _json[0]["processing"]["num_adjusted_elements"] = 32;
//        _json[0]["processing"]["num_adjusted_local_freq"] = 64;
//    } else {
//        _json[0]["processing"]["num_adjusted_elements"] = _json[0]["processing"]["num_elements"];
//        _json[0]["processing"]["num_adjusted_local_freq"] = _json[0]["processing"]["num_local_freq"];
//    }

/*
    // Generate number of blocks.
    int num_adj_elements = _json[0]["processing"]["num_adjusted_elements"].get<int>();
    int block_size = _json[0]["gpu"]["block_size"].get<int>();
    _json[0]["gpu"]["num_blocks"] = (int32_t)(num_adj_elements / block_size) *
        (num_adj_elements / block_size + 1) / 2.;
    int num_blocks = get_float("/gpu/num_blocks");
    (void)num_blocks;
*/
}

void Config::dump_config() {
    INFO("Config: %s", _json[0].dump().c_str());
}
