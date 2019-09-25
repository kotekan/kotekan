#include "Config.hpp"

#include "errors.h"
#include "visUtil.hpp"

#include "fmt.hpp"

#include <fstream>
#include <iostream>
#include <json.hpp>
#include <stdexcept>
#include <vector>

#ifdef WITH_SSL
#include <openssl/md5.h>
#endif

using std::vector;

namespace kotekan {

// Instantiation of the most common types to prevent them being built inline
// everywhere used.
template float Config::get(const string& base_path, const string& name);
template double Config::get(const string& base_path, const string& name);
template uint32_t Config::get(const string& base_path, const string& name);
template uint64_t Config::get(const string& base_path, const string& name);
template int32_t Config::get(const string& base_path, const string& name);
template int16_t Config::get(const string& base_path, const string& name);
template uint16_t Config::get(const string& base_path, const string& name);
template bool Config::get(const string& base_path, const string& name);
template std::string Config::get(const string& base_path, const string& name);
template std::vector<int32_t> Config::get(const string& base_path, const string& name);
template std::vector<uint32_t> Config::get(const string& base_path, const string& name);
template std::vector<float> Config::get(const string& base_path, const string& name);
template std::vector<std::string> Config::get(const string& base_path, const string& name);
template std::vector<nlohmann::json> Config::get(const string& base_path, const string& name);
template std::vector<std::complex<float>> Config::get(const string& base_path, const string& name);

Config::Config() {}

Config::~Config() {
    _json.clear();
}

void Config::parse_file(const string& file_name) {
    try {
        std::ifstream config_file_stream(file_name);
        config_file_stream >> _json;
    } catch (std::exception const& ex) {
        WARN_NON_OO("Could not parse json file: {:s}, error: {:s}", file_name, ex.what());
        throw ex;
    }
}

void Config::update_config(json updates) {
    _json = updates;
}

int32_t Config::num_links_per_gpu(const int32_t& gpu_id) {

    int32_t num_links = get<int32_t>("/", "num_links");
    vector<int32_t> link_map = get<std::vector<int32_t>>("/", "link_map");
    int32_t gpus_in_link = 0;

    for (int i = 0; i < num_links; ++i) {
        if (link_map[i] == gpu_id)
            gpus_in_link++;
    }
    return gpus_in_link;
}

json Config::get_value(const string& base_path, const string& name) {
    string search_path = base_path;
    for (;;) {

        if (search_path == "" && exists("/", name)) {
            json::json_pointer value_pointer(fmt::format(fmt("/{:s}"), name));
            return _json.at(value_pointer);
        }

        if (search_path == "")
            break;

        if (search_path == "/" && exists(search_path, name)) {
            json::json_pointer value_pointer(search_path + name);
            return _json.at(value_pointer);
        }

        if (exists(search_path, name)) {
            json::json_pointer value_pointer(fmt::format(fmt("{:s}/{:s}"), search_path, name));
            return _json.at(value_pointer);
        }

        std::size_t last_slash = search_path.find_last_of("/");
        search_path = search_path.substr(0, last_slash);
    }
    throw std::runtime_error(
        fmt::format(fmt("The config option: {:s} is required, but was not found in the path: {:s}"),
                    name, base_path));
}

bool Config::exists(const string& base_path, const string& name) {
    string search_path;
    if (base_path == "/") {
        search_path = base_path + name;
    } else {
        search_path = fmt::format(fmt("{:s}/{:s}"), base_path, name);
    }

    json::json_pointer search_pointer(search_path);
    try {
        _json.at(search_pointer);
    } catch (std::exception const& ex) {
        return false;
    }
    return true;
}

std::vector<json> Config::get_value(const std::string& name) const {
    std::vector<json> results;
    get_value_recursive(_json, name, results);
    return results;
}

void Config::get_value_recursive(const json& j, const std::string& name,
                                 std::vector<json>& results) const {
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == name)
            results.push_back(it.value());
        if (it->is_object())
            get_value_recursive(*it, name, results);
    }
}

void Config::dump_config() {
    INFO_NON_OO("Config: {:s}", _json.dump(4));
}

json& Config::get_full_config_json() {
    return _json;
}

#ifdef WITH_SSL
std::string Config::get_md5sum() {
    unsigned char md5sum[MD5_DIGEST_LENGTH];

    std::vector<std::uint8_t> v_msgpack = json::to_msgpack(_json);
    MD5((const unsigned char*)v_msgpack.data(), v_msgpack.size(), md5sum);

    char md5str[33];
    for (int i = 0; i < 16; i++)
        sprintf(&md5str[i * 2], "%02x", (unsigned int)md5sum[i]);

    return string(md5str);
}
#endif

} // namespace kotekan
