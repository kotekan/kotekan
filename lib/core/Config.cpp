#include "Config.hpp"

#include "fmt.hpp"  // for compile_string_to_view, format, fmt
#include "json.hpp" // for json, basic_json, iter_impl, operator>>

#include <cctype>    // for tolower
#include <cstddef>   // for size_t
#include <cstdint>   // for int32_t, uint8_t
#include <fstream>   // for basic_ifstream, basic_istream, ifstream
#include <map>       // for map
#include <mutex>     // for mutex, lock_guard
#include <set>       // for set
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for sprintf
#include <string>    // for string
#include <vector>    // for vector

#ifdef WITH_SSL
#include <openssl/md5.h> // for MD5, MD5_DIGEST_LENGTH
#endif

using nlohmann::json;
using std::vector;

namespace kotekan {

// Guards Config::_accessed for all instances. Access tracking is an opt-in
// diagnostic feature, so a single shared lock (rather than a per-instance
// mutex, which would make Config non-copyable) is sufficient.
static std::mutex access_lock;

// Instantiation of the most common types to prevent them being built inline
// everywhere used.
template float Config::get(const std::string& base_path, const std::string& name) const;
template double Config::get(const std::string& base_path, const std::string& name) const;
template unsigned char Config::get(const std::string& base_path, const std::string& name) const;
template unsigned short Config::get(const std::string& base_path, const std::string& name) const;
template unsigned int Config::get(const std::string& base_path, const std::string& name) const;
template unsigned long Config::get(const std::string& base_path, const std::string& name) const;
template unsigned long long Config::get(const std::string& base_path,
                                        const std::string& name) const;
template char Config::get(const std::string& base_path, const std::string& name) const;
template short Config::get(const std::string& base_path, const std::string& name) const;
template int Config::get(const std::string& base_path, const std::string& name) const;
template long Config::get(const std::string& base_path, const std::string& name) const;
template long long Config::get(const std::string& base_path, const std::string& name) const;
template bool Config::get(const std::string& base_path, const std::string& name) const;
template std::string Config::get(const std::string& base_path, const std::string& name) const;
template std::vector<unsigned char> Config::get(const std::string& base_path,
                                                const std::string& name) const;
template std::vector<unsigned short> Config::get(const std::string& base_path,
                                                 const std::string& name) const;
template std::vector<unsigned int> Config::get(const std::string& base_path,
                                               const std::string& name) const;
template std::vector<unsigned long> Config::get(const std::string& base_path,
                                                const std::string& name) const;
template std::vector<unsigned long long> Config::get(const std::string& base_path,
                                                     const std::string& name) const;
template std::vector<char> Config::get(const std::string& base_path, const std::string& name) const;
template std::vector<short> Config::get(const std::string& base_path,
                                        const std::string& name) const;
template std::vector<int> Config::get(const std::string& base_path, const std::string& name) const;
template std::vector<long> Config::get(const std::string& base_path, const std::string& name) const;
template std::vector<long long> Config::get(const std::string& base_path,
                                            const std::string& name) const;
template std::vector<std::string> Config::get(const std::string& base_path,
                                              const std::string& name) const;
template std::vector<nlohmann::json> Config::get(const std::string& base_path,
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

        // Resolve the value's full JSON pointer at the current search scope,
        // walking up the tree until found. Behaviour matches the original
        // chain of guards, but with a single exit so accesses can be recorded.
        std::string value_path;
        if (search_path == "" && exists("/", name)) {
            value_path = fmt::format(fmt("/{:s}"), name);
        } else if (search_path == "") {
            break;
        } else if (search_path == "/" && exists(search_path, name)) {
            value_path = search_path + name;
        } else if (exists(search_path, name)) {
            value_path = fmt::format(fmt("{:s}/{:s}"), search_path, name);
        } else {
            std::size_t last_slash = search_path.find_last_of("/");
            search_path = search_path.substr(0, last_slash);
            continue;
        }

        json::json_pointer value_pointer(value_path);
        const json& value = _json.at(value_pointer);
        // Note the access when tracking is enabled, attributed to the
        // requesting base path. A whole-object fetch records all its leaves.
        if (_usage_report_level != UsageReportLevel::off)
            record_access(value_path, value, base_path);
        return value;
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

void Config::set_usage_report_level(UsageReportLevel level) {
    _usage_report_level = level;
}

Config::UsageReportLevel Config::parse_usage_report_level(const json& value) {
    if (value.is_boolean())
        return value.get<bool>() ? UsageReportLevel::info : UsageReportLevel::off;
    if (value.is_string()) {
        std::string s = value.get<std::string>();
        for (char& c : s)
            c = std::tolower(static_cast<unsigned char>(c));
        if (s == "off" || s == "none" || s == "false")
            return UsageReportLevel::off;
        if (s == "info" || s == "true" || s == "on")
            return UsageReportLevel::info;
        if (s == "warn" || s == "warning")
            return UsageReportLevel::warn;
        if (s == "error")
            return UsageReportLevel::error;
        if (s == "fatal" || s == "fatal_error")
            return UsageReportLevel::fatal;
    }
    throw std::runtime_error(
        fmt::format(fmt("Invalid log_config_usage value {:s}; expected a boolean or one of "
                        "'off', 'info', 'warn', 'error', 'fatal_error'."),
                    value.dump()));
}

void Config::record_access(const std::string& path, const json& value,
                           const std::string& base_path) const {
    std::lock_guard<std::mutex> lock(access_lock);
    record_access_locked(path, value, base_path);
}

void Config::record_access_locked(const std::string& path, const json& value,
                                  const std::string& base_path) const {
    if (value.is_object()) {
        // The caller fetched a whole block (e.g. gpuProcess reading its
        // in_buffers/out_buffers map) and consumes every entry, so mark them all.
        for (auto it = value.begin(); it != value.end(); ++it)
            record_access_locked(fmt::format(fmt("{:s}/{:s}"), path, it.key()), it.value(),
                                 base_path);
    } else {
        _accessed[path].insert(base_path);
    }
}

// Leaves that are never read through get(): the framework dispatch markers
// (kotekan_buffer/kotekan_stage/kotekan_metadata_pool/kotekan_update_endpoint),
// which the factories and configUpdater consume by walking get_full_config_json()
// directly, plus the top-level meta keys. These are not tunable config values,
// so excluding them keeps the usage summary focused on real settings.
static bool is_framework_key(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    const std::string key = (slash == std::string::npos) ? path : path.substr(slash + 1);
    if (key.rfind("kotekan_", 0) == 0)
        return true;
    return path == "/type" || path == "/log_config_usage";
}

void Config::collect_leaf_paths(const json& j, const std::string& path,
                                vector<std::string>& leaves) const {
    if (j.is_object()) {
        // configUpdater-managed blocks are consumed via the update endpoint
        // (a get_full_config_json() tree walk), not through get(), so their
        // values never register as accessed -- don't count them.
        if (j.contains("kotekan_update_endpoint"))
            return;
        for (auto it = j.begin(); it != j.end(); ++it)
            collect_leaf_paths(it.value(), fmt::format(fmt("{:s}/{:s}"), path, it.key()), leaves);
    } else if (!is_framework_key(path)) {
        leaves.push_back(path);
    }
}

void Config::log_access_summary() const {
    if (_usage_report_level == UsageReportLevel::off)
        return;

    // Enumerate every leaf in the config, then split into accessed vs. not,
    // using the paths recorded by get_value().
    vector<std::string> leaves;
    collect_leaf_paths(_json, "", leaves);

    std::string accessed;
    std::string not_accessed;
    std::size_t n_accessed = 0;

    {
        std::lock_guard<std::mutex> lock(access_lock);
        for (const std::string& leaf : leaves) {
            auto it = _accessed.find(leaf);
            if (it == _accessed.end()) {
                not_accessed += fmt::format(fmt("\n    {:s}"), leaf);
                continue;
            }
            n_accessed++;
            std::string base_paths;
            for (const std::string& base_path : it->second) {
                if (!base_paths.empty())
                    base_paths += ", ";
                base_paths += base_path;
            }
            accessed += fmt::format(fmt("\n    {:s}  <- {:s}"), leaf, base_paths);
        }
    }

    const std::size_t n_unused = leaves.size() - n_accessed;
    const std::string summary =
        fmt::format(fmt("Config usage summary: {:d} of {:d} leaf items accessed.\n"
                        "  Accessed (item <- requesting paths):{:s}\n"
                        "  Not accessed:{:s}"),
                    n_accessed, leaves.size(), accessed, not_accessed);

    // With nothing unused there is nothing to flag, regardless of level.
    if (n_unused == 0) {
        INFO_NON_OO("{:s}", summary);
        return;
    }

    switch (_usage_report_level) {
        case UsageReportLevel::warn:
            WARN_NON_OO("{:s}", summary);
            break;
        case UsageReportLevel::error:
            ERROR_NON_OO("{:s}", summary);
            break;
        case UsageReportLevel::fatal:
            ERROR_NON_OO("{:s}", summary);
            // Note: not FATAL_ERROR_NON_OO, which throws -- this runs from
            // ~kotekanMode (a noexcept destructor). Set the error message and
            // exit code directly so kotekan exits non-zero after teardown.
            kotekanLogging::set_error_message(
                fmt("Config usage: {:d} config item(s) were not accessed."), n_unused);
            exit_kotekan(ReturnCode::FATAL_ERROR);
            break;
        case UsageReportLevel::info:
        case UsageReportLevel::off:
        default:
            INFO_NON_OO("{:s}", summary);
            break;
    }
}

const json& Config::get_full_config_json() const {
    return _json;
}

#ifdef WITH_SSL
std::string Config::get_md5sum() const {
    unsigned char md5sum[MD5_DIGEST_LENGTH];

    vector<std::uint8_t> v_msgpack = json::to_msgpack(_json);

    // The MD5 function is deprecated in openssl 3.0, but we want to
    // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    MD5((const unsigned char*)v_msgpack.data(), v_msgpack.size(), md5sum);
#pragma GCC diagnostic pop

    char md5str[33];
    for (int i = 0; i < 16; i++)
        sprintf(&md5str[i * 2], "%02x", (unsigned int)md5sum[i]);

    return std::string(md5str);
}
#endif

} // namespace kotekan
