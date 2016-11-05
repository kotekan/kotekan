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

int32_t Config::get_int(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_number_integer() && !_json[0][jp].is_number_float() ) {
        throw std::runtime_error("The value " + pointer + " isn't an integer or doesn't exist");
    }
    return _json[0][jp].get<int32_t>();
}

double Config::get_double(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_number_float()) {
        throw std::runtime_error("The value " + pointer + " isn't a double or doesn't exist");
    }
    return _json[0][jp].get<double>();
}

float Config::get_float(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_number_float()) {
        throw std::runtime_error("The value " + pointer + " isn't a float or doesn't exist");
    }
    return _json[0][jp].get<float>();
}

string Config::get_string(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_string()) {
        throw std::runtime_error("The value " + pointer + " isn't a string or doesn't exist");
    }
    return _json[0][jp].get<string>();
}

bool Config::get_bool(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_boolean()) {
        throw std::runtime_error("The value " + pointer + " isn't a boolean or doesn't exist");
    }
    return _json[0][jp].get<bool>();
}

vector<int32_t> Config::get_int_array(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_array()) {
        throw std::runtime_error("The value " + pointer + " isn't an array or doesn't exist");
    }
    return _json[0][jp].get< vector<int32_t> >();
}

vector<double> Config::get_double_array(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_array()) {
        throw std::runtime_error("The value " + pointer + " isn't an array or doesn't exist");
    }
    return _json[0][jp].get< vector<double> >();
}

vector<float> Config::get_float_array(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_array()) {
        throw std::runtime_error("The value " + pointer + " isn't an array or doesn't exist");
    }
    return _json[0][jp].get< vector<float> >();
}

vector<string> Config::get_string_array(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_array()) {
        throw std::runtime_error("The value " + pointer + " isn't an array or doesn't exist");
    }
    return _json[0][jp].get< vector<string> >();
}

vector<json> Config::get_json_array(const string& pointer) {
    json::json_pointer jp(pointer);

    if (!_json[0][jp].is_array()) {
        throw std::runtime_error("The value " + pointer + " isn't an array or doesn't exist");
    }

    return _json[0][jp].get< vector<json> >();
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

    int32_t num_links = get_int("/fpga_network/num_links");
    vector<int32_t> link_map = get_int_array("/gpu/link_map");
    int32_t gpus_in_link = 0;

    for (int i = 0; i < num_links; ++i) {
        if (link_map[i] == gpu_id)
            gpus_in_link++;
    }
    return gpus_in_link;
}

void Config::generate_extra_options() {

    // Create the inverse remap
    vector<int32_t> remap = get_int_array("/processing/product_remap");
    vector<int32_t> inverse_remap;
    inverse_remap.resize(remap.size());

    // Given a channel ID, where is it in FPGA order.
    for(uint32_t i = 0; i < remap.size(); ++i) {
        inverse_remap[remap[i]] = i;
    }
    _json[0]["processing"]["inverse_product_remap"] = inverse_remap;

    // Special case for 16-element version
    if (_json[0]["processing"]["num_elements"].get<int>() < 32) {
        _json[0]["processing"]["num_adjusted_elements"] = 32;
        _json[0]["processing"]["num_adjusted_local_freq"] = 64;
    } else {
        _json[0]["processing"]["num_adjusted_elements"] = _json[0]["processing"]["num_elements"];
        _json[0]["processing"]["num_adjusted_local_freq"] = _json[0]["processing"]["num_local_freq"];
    }

    // Generate number of blocks.
    int num_adj_elements = _json[0]["processing"]["num_adjusted_elements"].get<int>();
    int block_size = _json[0]["gpu"]["block_size"].get<int>();
    _json[0]["gpu"]["num_blocks"] = (int32_t)(num_adj_elements / block_size) *
        (num_adj_elements / block_size + 1) / 2.;
    int num_blocks = get_float("/gpu/num_blocks");
    (void)num_blocks;
}

void Config::dump_config() {
    INFO("Config: %s", _json[0].dump().c_str());
}
