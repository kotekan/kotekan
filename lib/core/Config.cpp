#include "Config.hpp"

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, iter_impl, basic_json<>::object_t, operator>>, basic_json

#include <cstdint>   // for int32_t
#include <fstream>   // for ifstream, istream, size_t
#include <map>       // for map<>::key_type
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for sprintf
#include <vector>    // for vector

#ifdef WITH_SSL
#include <openssl/md5.h> // for MD5, MD5_DIGEST_LENGTH
#endif

using nlohmann::json;
using std::vector;

namespace kotekan {

// Instantiation of the most common types to prevent them being built inline
// everywhere used.
template float Config::get(const std::string& base_path, const std::string& name) const;
template double Config::get(const std::string& base_path, const std::string& name) const;
template uint32_t Config::get(const std::string& base_path, const std::string& name) const;
template uint64_t Config::get(const std::string& base_path, const std::string& name) const;
template int32_t Config::get(const std::string& base_path, const std::string& name) const;
template int16_t Config::get(const std::string& base_path, const std::string& name) const;
template uint16_t Config::get(const std::string& base_path, const std::string& name) const;
template bool Config::get(const std::string& base_path, const std::string& name) const;
template std::string Config::get(const std::string& base_path, const std::string& name) const;
template vector<int32_t> Config::get(const std::string& base_path, const std::string& name) const;
template vector<uint32_t> Config::get(const std::string& base_path, const std::string& name) const;
template vector<float> Config::get(const std::string& base_path, const std::string& name) const;
template vector<std::string> Config::get(const std::string& base_path,
                                         const std::string& name) const;
template vector<nlohmann::json> Config::get(const std::string& base_path,
                                            const std::string& name) const;

Config::Config() {}

Config::~Config() {
    _json.clear();
}

void Config::parse_file(const std::string& file_name) {
    try {
        std::ifstream config_file_stream(file_name);
        config_file_stream >> _json;
    } catch (std::exception const& ex) {
        WARN_NON_OO("Could not parse json file: {:s}, error: {:s}", file_name, ex.what());
        throw;
    }
}

void Config::update_config(json updates) {
    _json = updates;
}

int32_t Config::num_links_per_gpu(const int32_t& gpu_id) const {

    int32_t num_links = get<int32_t>("/", "num_links");
    vector<int32_t> link_map = get<vector<int32_t>>("/", "link_map");
    int32_t gpus_in_link = 0;

    for (int i = 0; i < num_links; ++i) {
        if (link_map[i] == gpu_id)
            gpus_in_link++;
    }
    return gpus_in_link;
}

json Config::get_value(const std::string& base_path, const std::string& name) const {
    std::string search_path = base_path;
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

bool Config::exists(const std::string& base_path, const std::string& name) const {
    std::string search_path;
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

vector<json> Config::get_value(const std::string& name) const {
    vector<json> results;
    get_value_recursive(_json, name, results);
    return results;
}

void Config::get_value_recursive(const json& j, const std::string& name,
                                 vector<json>& results) const {
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == name)
            results.push_back(it.value());
        if (it->is_object())
            get_value_recursive(*it, name, results);
    }
}

void Config::dump_config() const {
    INFO_NON_OO("Config: {:s}", _json.dump(4));
}

const json& Config::get_full_config_json() const {
    return _json;
}

#ifdef WITH_SSL
std::string Config::get_md5sum() const {
    unsigned char md5sum[MD5_DIGEST_LENGTH];

    vector<std::uint8_t> v_msgpack = json::to_msgpack(_json);
    MD5((const unsigned char*)v_msgpack.data(), v_msgpack.size(), md5sum);

    char md5str[33];
    for (int i = 0; i < 16; i++)
        sprintf(&md5str[i * 2], "%02x", (unsigned int)md5sum[i]);

    return std::string(md5str);
}
#endif

} // namespace kotekan
