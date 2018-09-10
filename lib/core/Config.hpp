#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <vector>
#include <cxxabi.h>

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
    template<typename T>
    T get(const string& base_path, const string& name);

    // Same as get_int, but if it cannot find the value, it returns `default_value`
    template<typename T>
    T get_default(const string& base_path, const string& name, T default_value);

    // Treats the value as an arithmetic expression with other variables,
    // and return the result of evaluating it.
    int32_t get_int_eval(const string& base_path, const string& name);

    // Treats the value as an arithmetic expression with other variables,
    // and return the result of evaluating it.
    double get_double_eval(const string& base_path, const string& name);

    // Returns true if the path exists
    bool exists(const string& base_path, const string& name);

    vector<int32_t> get_int_array(const string& base_path, const string& name);
    vector<int32_t> get_int_array_default(const string& base_path, const string& name, vector<int32_t> default_value);
    vector<float> get_float_array(const string& base_path, const string& name);
    vector<float> get_float_array_default(const string& base_path, const string& name, vector<float> default_value);
    vector<double> get_double_array(const string& base_path, const string& name);
    vector<string> get_string_array(const string& base_path, const string& name);
    vector<json> get_json_array(const string& base_path, const string& name);

    template<typename T>
    vector<T> get_array(const string& base_path, const string& name);

    void parse_file(const string &file_name);

    // @param updates Json object with values to be replaced.
    // @param start_fpga_seq The fpga seq number to update the config on.
    // This value must be in the future.
    void update_config(json updates);

    // This function should be moved, it doesn't really belong here...
    int32_t num_links_per_gpu(const int32_t &gpu_id);

    // @breaf Finds the value with key "name" starts looking at the
    // "base_pointer" location, and then works backwards up the config tree.
    // @param base_pointer Contains a JSON pointer which points to the
    // process's location in the config tree. i.e. /vdif_cap/disk_write
    // @param name The name of the property i.e. num_frequencies
    json get_value(const string &base_pointer, const string &name);

    /**
     * @brief Updates a config value at an existing config option
     *
     * Only accepts updates to known config locations, and should only be used
     * to update config within a processes' own @c unique_name path.
     *
     * @todo This currently will not update inherited config options.
     *       So any updatable config options must be given in the process block.
     *       The way around this is likely to copy inherited, either on access or
     *       on update with this function.
     *
     * @todo Currently this will take any type that can translate to json,
     *       we should likely make this a bit more restrictive and/or require
     *       the update type matches the base value type already in the config.
     *
     * @param base_path The unique name of the process, or "/" for the root process
     * @param name The name of the value
     * @param value The value to assign to the json pointer formed by base_path/name
     */
    template <typename T>
    void update_value(const string &base_path, const string &name, const T &value);

#ifdef WITH_SSL
    /**
     * @brief Generate an MD5 hash of the config
     *
     * The HASH is based on the json string dump with no spaces or newlines
     *
     * Only avaibile if OpenSSL was installed on a system,
     * so wrap any uses in @c #ifdef WITH_SSL
     *
     * @return The MD5sum as 32 char hex std::string
     */
    std::string get_md5sum();
#endif

    // Returns the full json data structure (for internal framework use)
    json &get_full_config_json();

    // Debug
    void dump_config();
private:

    json _json;

};

template <typename T>
void Config::update_value(const string &base_path, const string &name, const T &value) {
    string update_path = base_path + "/" + name;
    json::json_pointer path(update_path);

    try {
        _json.at(path) = value;
    } catch (std::exception const & ex) {
        throw std::runtime_error("Failed to update config value at: " + update_path + " message: " + ex.what());
    }
}

template<typename T>
T Config::get(const string& base_path, const string& name) {
    json json_value;
    T value;
    try {
        json_value = get_value(base_path, name);
        value = json_value.get<T>();
    } catch (std::exception const & ex) {
        int status;
        throw std::runtime_error(
                    "The value " + name + " in path " + base_path
                    + " is not of type '" +
                    abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status)
                    + "' or doesn't exist");
    }

    return value;
}

template<typename T>
T Config::get_default(const string& base_path, const string& name,
                         T default_value) {
    try {
        T value = get<T>(base_path, name);
        return value;
    } catch (std::exception const & ex) {
        return default_value;
    }
}

template<typename T>
vector<T> Config::get_array(const string& base_path, const string& name)
{
    json value = get_value(base_path, name);

    if (!value.is_array()) {
        throw std::runtime_error("The value " + name + " in path " + base_path + " isn't an array or doesn't exist");
    }
    return value.get<vector<T>>();
}

#endif /* CONFIG_HPP */

