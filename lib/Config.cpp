#include "Config.hpp"
#include "errors.h"
#include "configEval.hpp"

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

int32_t Config::get_int_default(const string& base_path, const string& name, int32_t default_value) {
    try {
        int32_t value = get_int(base_path, name);
        return value;
    } catch (std::exception const & ex) {
        return default_value;
    }
}

int32_t Config::get_int_eval(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!(value.is_string() || value.is_number())) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a number or string to eval or doesn't exist");
    }
    return eval_compute_int64(*this, base_path, value.get<string>());
}

double Config::get_double(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_number()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a double or doesn't exist");
    }
    return value.get<double>();
}

double Config::get_double_default(const string& base_path, const string& name, double default_value) {
    try {
        double value = get_double(base_path, name);
        return value;
    }  catch (std::exception const & ex) {
        return default_value;
    }
}

double Config::get_double_eval(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!(value.is_string() || value.is_number())) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a number or string to eval or doesn't exist");
    }
    return eval_compute_double(*this, base_path, value.get<string>());
}

float Config::get_float(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_number_float()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a float or doesn't exist");
    }
    return value.get<float>();
}

float Config::get_float_default(const string& base_path, const string& name, float default_value) {
    try {
        float value = get_float(base_path, name);
        return value;
    }  catch (std::exception const & ex) {
        return default_value;
    }
}

string Config::get_string(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_string()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a string or doesn't exist");
    }
    return value.get<string>();
}

string Config::get_string_default(const string& base_path, const string& name, const string& default_value) {
    try {
        string value = get_string(base_path, name);
        return value;
    }  catch (std::exception const & ex) {
        return default_value;
    }
}

bool Config::get_bool(const string& base_path, const string& name) {
    json value = get_value(base_path, name);

    if (!value.is_boolean()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't a boolean or doesn't exist");
    }
    return value.get<bool>();
}

bool Config::get_bool_default(const string& base_path, const string& name, bool default_value) {
    try {
        bool value = get_bool(base_path, name);
        return value;
    }  catch (std::exception const & ex) {
        return default_value;
    }
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

void Config::dump_config() {
    INFO("Config: %s", _json[0].dump().c_str());
}

json &Config::get_full_config_json() {
    return _json[0];
}
