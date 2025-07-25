#ifndef CONFIGMTRACKER_H
#define CONFIGMTRACKER_H

#include "Config.hpp"     // for Config
#include "Stage.hpp"      // for Stage
#include "restServer.hpp" // for connectionInstance
#include "restClient.hpp" // for restClient::restReply, restClient
#include "json.hpp" // for json

#include <openssl/md5.h>
#include <sstream>
#include <iomanip>
#include <string>     // for string
#include <sys/stat.h> // for stat
#include <map>        // for map
#include <mutex>      // for mutex

namespace kotekan {

/**
 * @class ConfigTracker
 * @brief Kotekan core component that tracks the (startup-time) configurations through a pipeline.
 *
 * This class must be registered with a kotekan REST server instance by
 * using the @c register_with_server() function.
 *
 * This class is a singleton, and can be accessed with @c instance()
 */
class ConfigTracker {
public:
    /**
     * @brief Get the global ConfigTracker.
     *
     * @returns A reference to the global ConfigTracker instance.
     **/
    static ConfigTracker& instance() {
        static ConfigTracker instance;
        return instance;
    }

    // Remove the implicit copy/assignments to prevent copying
    ConfigTracker(const ConfigTracker&) = delete;
    void operator=(const ConfigTracker&) = delete;

    /**
     * @brief Struct to hold a (host, port) pair.
     */
    struct HostPort {
        std::string host;
        uint16_t port;

        bool operator==(const HostPort& other) const {
            return host == other.host && port == other.port;
        }
        // < operator for std::map sorting
        bool operator<(const HostPort& other) const {
            return std::tie(host, port) < std::tie(other.host, other.port);
        }
    };

    /**
     * @brief Struct to hold information about a configuration.
     *
     * This struct is used to store the hash, software version information, and the
     * configuration JSON object.
     */
    struct ConfigInfo {
        nlohmann::json config;
        std::string json_hash;
        
        std::string kotekan_version;
        std::string kotekan_build_branch;
        std::string kotekan_git_commit_hash;
        std::string kotekan_cmake_options;

        // Default constructor
        ConfigInfo() = default;

        // Constructor with all parameters
        ConfigInfo(const nlohmann::json& config, 
                    const std::string& json_hash,
                    const std::string& kotekan_version,
                    const std::string& kotekan_build_branch,
                    const std::string& kotekan_git_commit_hash,
                    const std::string& kotekan_cmake_options)
            : config(config), json_hash(json_hash), 
            kotekan_version(kotekan_version),
            kotekan_build_branch(kotekan_build_branch),
            kotekan_git_commit_hash(kotekan_git_commit_hash),
            kotekan_cmake_options(kotekan_cmake_options) {}

        // Constructor from JSON
        explicit ConfigInfo(const nlohmann::json& j) {
            config = j.at("config");
            json_hash = j.at("json_hash");
            kotekan_version = j.at("kotekan_version");
            kotekan_build_branch = j.at("kotekan_build_branch");
            kotekan_git_commit_hash = j.at("kotekan_git_commit_hash");
            kotekan_cmake_options = j.at("kotekan_cmake_options");
        }
        
        // Convert to JSON
        nlohmann::json to_json() const {
            return nlohmann::json{
                {"config", config},
                {"json_hash", json_hash},
                {"kotekan_version", kotekan_version},
                {"kotekan_build_branch", kotekan_build_branch},
                {"kotekan_git_commit_hash", kotekan_git_commit_hash},
                {"kotekan_cmake_options", kotekan_cmake_options}
            };
        }
    };

    /**
     * @brief Check if the number of configs and hashes are consistent.
     *
     * This function checks if the number of configurations stored in _configs
     * matches the number of hashes stored in _config_hashes.
     *
     * @returns True if the sizes are consistent, false otherwise.
     */
    bool check_num_configs_consistent() const {
        std::lock_guard<std::mutex> lock(_lock);
        // _configs and _config_hashes should always have the same size

        if (_configs.size() != _config_hashes.size()) {
            FATAL_ERROR_NON_OO("ConfigTracker: _configs and _config_hashes have different sizes: {} vs {}",
                               _configs.size(), _config_hashes.size());
            return false;
        }

        return true;
    }

    /**
     * @brief Get the number of configurations stored in the tracker.
     * @returns The number of configurations.
     */
    std::size_t n_configs() const {
        check_num_configs_consistent();

        std::lock_guard<std::mutex> lock(_lock);
        return _configs.size(); 
    }

    /**
     * @brief Get the canonical hash of a (jsonified) config.
     *
     * This function generates a consistent hash for a given configuration JSON object.
     * The JSON object should have any "updatable_config" fields stripped before hashing.
     * (This function only checks for that, it does not strip them itself.)
     *
     * @param filtered_json The configuration JSON object to hash.
     * @returns The canonical hash as a string.
     */
    std::string jsonHash(const nlohmann::json& filtered_json) const {
        std::stringstream ss;

        // nlohmann::json::dump() uses an alpha-ordered map for objects, so the
        // config should be serialized in a consistent order.

        // In order for this to hash configs correctly, this assumes the updatable_config
        // field is removed, and versioning information has been added.
        if( filtered_json.contains("updatable_config")) {
            FATAL_ERROR_NON_OO("ConfigTracker: jsonHash called with updatable_config field present.");
        }

        // Stick to a string dump; less likely to run into floating point issues?
        ss << filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);

        std::string serialized = ss.str();
        unsigned char md5_result[MD5_DIGEST_LENGTH];

        // The MD5 function is deprecated in openssl 3.0, but we want to
        // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MD5(reinterpret_cast<const unsigned char*>(serialized.c_str()), serialized.size(), md5_result);
#pragma GCC diagnostic pop


        std::stringstream md5_ss;
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
            md5_ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5_result[i]);

        return md5_ss.str();
    }

    /**
     * @brief Set a hash of the combined hash of all configurations stored in the tracker.
     * i.e. a hash representing the current tracker state.
     */
    void setTrackerHash() {

        std::stringstream ss;
        {
            std::lock_guard<std::mutex> lock(_lock);
            for (const auto& [hash, _] : _config_hashes) {
                ss << hash;
            }
        }

        // Get hash of the concatenated hashes

        unsigned char md5_result[MD5_DIGEST_LENGTH];

        // The MD5 function is deprecated in openssl 3.0, but we want to
        // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MD5(reinterpret_cast<const unsigned char*>(ss.str().c_str()), ss.str().size(), md5_result);
#pragma GCC diagnostic pop

        // Convert the MD5 result to a hex string
        std::stringstream md5_ss;
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
            md5_ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5_result[i]);
        {
            std::lock_guard<std::mutex> lock(_lock);
            // Store the combined hash
            _tracker_hash = md5_ss.str();
        }
        
        DEBUG_NON_OO("Combined hash set to: {}", _tracker_hash);
    }

    /**
     * @brief Get a hash representation of all configurations stored in the tracker.
     *
     * @returns A string hash of the combined hash of all configurations, i.e. a hash
     * representing the current tracker state.
     */
    std::string getTrackerHash() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _tracker_hash;
    }
    
    /**
     * @brief Insert JSON configuration into the tracker.
     * This function checks requires "updatable_config" fields have been removed.
     * This is an overload that allows using a ConfigInfo object directly.
     * 
     * @param host The host where the kotekan instance with the configuration is running.
     * @param port The port where the kotekan instance with the configuration is running.
     * @param config_info The configuration information to insert.
     * @throws Error if a config with the same host and port already exists
     */
    void insertConfig(std::string host,
                      uint16_t port,
                      ConfigInfo config_info) {

        // Make sure the config_info doesn't have an "updatable_config" field in its config
        if (config_info.config.contains("updatable_config")) {
            FATAL_ERROR_NON_OO("ConfigTracker: insertConfig called with updatable_config field present.");
        }

        HostPort host_port{host, port};

        // Store the config in the map with its hash
        {
            std::lock_guard<std::mutex> lock(_lock);

            if (_configs.count(host_port)) {
                // If a config already exists with the same host and port,
                // make sure the hash and version metadata match.
                if( _configs[host_port].json_hash != config_info.json_hash ||
                    _configs[host_port].kotekan_version != config_info.kotekan_version ||
                    _configs[host_port].kotekan_build_branch != config_info.kotekan_build_branch ||
                    _configs[host_port].kotekan_git_commit_hash != config_info.kotekan_git_commit_hash ||
                    _configs[host_port].kotekan_cmake_options != config_info.kotekan_cmake_options) {
                    FATAL_ERROR_NON_OO("ConfigTracker: conflicting configuration data present for host: {}, port: {}",
                                  host, port);
                    return;
                }
                return;
            }

            _configs.emplace(host_port, config_info);
            _config_hashes.emplace(config_info.json_hash, host_port);
        }
        // Sanity check to ensure that the number of configs is consistent
        check_num_configs_consistent();

        // Update the combined hash
        setTrackerHash();
        DEBUG_NON_OO("ConfigTracker: inserted config for host: {}, port: {}, hash: {}",
                    host, port, config_info.json_hash);
    }

    /**
     * @brief Insert JSON configuration into the tracker.
     * Creates a ConfigInfo object from the parameters.
     * This function will strip "updatable_config" fields.
     * 
     * @param host The host where the kotekan instance with the configuration is running.
     * @param port The port where the kotekan instance with the configuration is running.
     * @param config_json The JSON configuration to insert.
     * @param kotekan_version The version of Kotekan.
     * @param kotekan_build_branch The build branch of Kotekan.
     * @param kotekan_git_commit_hash The git commit hash of Kotekan.
     * @param kotekan_cmake_options The CMake options used to build Kotekan.
     */
    void insertConfig(std::string host,
                      uint16_t port,
                      const nlohmann::json& config_json,
                      const std::string& kotekan_version,
                      const std::string& kotekan_build_branch,
                      const std::string& kotekan_git_commit_hash,
                      const std::string& kotekan_cmake_options) {
        // Strip updatable config fields before hashing
        nlohmann::json filtered_json;
        for (auto& [key, value] : config_json.items()) {
            if (key != "updatable_config") {
                filtered_json[key] = value;
            }
        }

        std::string json_hash = jsonHash(filtered_json);

        ConfigInfo info = ConfigInfo(filtered_json, json_hash, kotekan_version,
            kotekan_build_branch, kotekan_git_commit_hash, kotekan_cmake_options);
        
        // Call insertConfig with the constructed ConfigInfo
        insertConfig(host, port, info);
    }

    /**
     * @brief Determine if a config exists in the _configs map.
     *
     * @param hash The hash of the config.
     * @returns True if the config exists, false otherwise.
     */
    bool hasConfig(std::string host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _configs.count({host, port}) != 0;
    }

    /**
     * @brief check hasConfig using a hash string instead of host and port.
     */
    bool hasConfig(std::string hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _config_hashes.count(hash) != 0;
    }

    /**
     * @brief A callback function for the REST server to use.
     * This function returns a configuration corresponding to a given hash,
     * or all configurations if no hash is provided.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_configs_callback(connectionInstance& conn) {
        check_num_configs_consistent();

        nlohmann::json return_json = {};

        auto query_args = conn.get_query();

        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& config : _configs) {
            // If a hash is provided, only return the config with that hash
            if (query_args.find("hash") != query_args.end()) {
                std::string hash = query_args["hash"];
                if (!hash.empty() && config.second.json_hash != hash) {
                    continue;
                }
            }

            // Serialize the ConfigInformation into JSON
            const auto& host_port = config.first;
            const auto& info = config.second;

            // Concatenate host:port in a string
            std::string host_port_str = host_port.host + ":" + std::to_string(host_port.port);
            return_json[host_port_str] = nlohmann::json::object();

            // Add the config JSON and metadata to the return JSON
            return_json[host_port_str]["config"] = info.config;
            return_json[host_port_str]["json_hash"] = info.json_hash;
            return_json[host_port_str]["kotekan_version"] = info.kotekan_version;
            return_json[host_port_str]["kotekan_build_branch"] = info.kotekan_build_branch;
            return_json[host_port_str]["kotekan_git_commit_hash"] = info.kotekan_git_commit_hash;
            return_json[host_port_str]["kotekan_cmake_options"] = info.kotekan_cmake_options;
        }

        conn.send_json_reply(return_json);
    }

    /**
     * @brief A callback function for the REST server to use.
     * This function returns a json object containing the hashes of all configurations.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_hashes_callback(connectionInstance& conn) {
        check_num_configs_consistent();

        nlohmann::json return_json = {};
        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& hash : _config_hashes) {
            // Add the hash and its corresponding host:port to the return JSON
            return_json[hash.first] = nlohmann::json::object();
            return_json[hash.first]["host"] = hash.second.host;
            return_json[hash.first]["port"] = hash.second.port;
        }
        conn.send_json_reply(return_json);
    }

    /**
     * @brief Registers this class with the REST server, creating the
     *        /trackers end point
     * If a hash is provided, it will return the config for that hash.
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server) {
        using namespace std::placeholders;
        // register callback for /config_tracker, pass along hash if provided
        rest_server->register_get_callback("/config_tracker_configs",
                                           std::bind(&ConfigTracker::trackers_configs_callback, this, _1));
        rest_server->register_get_callback("/config_tracker_hashes",
                                           std::bind(&ConfigTracker::trackers_hashes_callback, this, _1));
    }

    /**
     * @brief Request and insert upstream configs into the tracker.
     * This function will fetch hashes from an upstream REST server, then:
     *   1) check if upstream configs are already present locally, if not, fetch them.
     *   2) If the hash already exists, check to make sure the (hash, host:port) hasn't changed (otherwise fail).
     */
    void getUpstreamConfigs(const std::string& host,
                                uint16_t port) {


        // Send a request to the upstream server to get all config hashes.
        nlohmann::json request_json = {};
        restClient::restReply reply = restClient::instance().make_request_blocking("/config_tracker_hashes",
                                    {}, host, port, 1, -1);
        // reply is a pair with success boolean and the reply string
        if (!reply.first) {
            ERROR_NON_OO("Failed to get config hashes from upstream host: {}, port: {}", host, port);
            return;
        }

        // Check if the response contains any hashes
        if (reply.second.empty()) {
            ERROR_NON_OO("No configs found at upstream host: {}, port: {}", host, port);
            return;
        }

        // Convert the reply string to a JSON object
        nlohmann::json response_json;
        try {
            response_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            ERROR_NON_OO("Failed to parse JSON response from upstream host: {}, port: {}. Error: {}",
                          host, port, e.what());
            return;
        }
        
        // Iterate over the hashes and fetch each config
        for (const auto& item : response_json.items()) {

            const std::string& hash = item.key();
            const nlohmann::json& host_port_json = item.value();
            std::string upstream_host = host_port_json.value("host", host);
            uint16_t upstream_port = host_port_json.value("port", port);

            // Check if the config already exists in the tracker
            if (hasConfig(hash)) {
                std::lock_guard<std::mutex> lock(_lock);

                // If it exists, make sure the host and port match
                if(_config_hashes[hash].host == upstream_host &&
                   _config_hashes[hash].port == upstream_port) {
                    // Config already exists with the same host and port, continue
                    continue;
                } else {
                    FATAL_ERROR_NON_OO("Hash conflict for {}, upstream host: {}, port: {}",
                                  hash, upstream_host, upstream_port);
                }
            }

            // If it doesn't exist, fetch the config from the upstream server
            request_json = {{"hash", hash}};
            reply = restClient::instance().make_request_blocking("/config_tracker_configs",
                                    request_json, host, port, 1, -1);
            // Check if the request was successful
            if (!reply.first) { 
                ERROR_NON_OO("Failed to get config for hash: {} from upstream host: {}, port: {}",
                              hash, upstream_host, upstream_port);
                continue;
            }
            // Parse the response JSON
            nlohmann::json config_response_json;
            try {
                config_response_json = nlohmann::json::parse(reply.second);
            } catch (const nlohmann::json::parse_error& e) {
                ERROR_NON_OO("Failed to parse JSON response for hash: {} from upstream host: {}, port: {}. Error: {}",
                              hash, upstream_host, upstream_port, e.what());
                continue;
            }

            // Check if the response contains the config
            std::string host_port_str = host + ":" + std::to_string(port);
            if (config_response_json.contains(host_port_str)) {
                ConfigInfo info = ConfigInfo(config_response_json[host_port_str]);
                insertConfig(host, port, info);
            } else {
                // If the config was not found, log an error or take appropriate action
                ERROR_NON_OO("Config not found for hash: {}", hash);
            }
        }
        // Sanity check to ensure that the number of configs is consistent
        check_num_configs_consistent();
    }

    /**
     * @brief Write all configuration data to disk with detailed error reporting.
     * 
     * @param directory The directory to write files to
     * @return Number of configurations successfully written
     */
    size_t writeConfigsToDisk(const std::string& directory) const {
        // Check if directory exists and is writable
        struct stat info;
        if (stat(directory.c_str(), &info) != 0) {
            std::string err = "Directory does not exist: " + directory;
            throw std::runtime_error(err);
        }
        if (!(info.st_mode & S_IFDIR)) {
            std::string err = "Path is not a directory: " + directory;
            throw std::runtime_error(err);
        }
        
        std::lock_guard<std::mutex> lock(_lock);
        size_t written = 0;
        
        for (const auto& [host_port, config_info] : _configs) {
            // Create filename - use underscore instead of colon for portability
            std::ostringstream filename_stream;
            filename_stream << directory << "/" 
                        << host_port.host << "_" 
                        << host_port.port << ".json";
            std::string filename = filename_stream.str();
            
            try {
                // Write atomically by writing to temp file first
                std::string temp_filename = filename + ".tmp";
                
                {
                    std::ofstream file(temp_filename);
                    if (!file) {
                        throw std::runtime_error("Cannot open file for writing");
                    }
                    
                    // Write with pretty formatting
                    file << config_info.to_json().dump(4, ' ', false, 
                                                    nlohmann::json::error_handler_t::strict);
                    
                    if (!file.good()) {
                        throw std::runtime_error("Write failed");
                    }
                }  // file closed here
                
                // Atomically rename temp file to final name
                if (std::rename(temp_filename.c_str(), filename.c_str()) != 0) {
                    throw std::runtime_error("Failed to rename temp file");
                }
                
                ++written;
                
            } catch (const std::exception& e) {
                std::string err = "Error writing " + filename + ": " + e.what();
                ERROR_NON_OO("{}", err);
                
                // Clean up temp file if it exists
                std::remove((filename + ".tmp").c_str());
            }
        }
        
        return written;
    }


private:
    /// Constructor, we don't want anyone to call this
    ConfigTracker() = default;

    /// List of (host:port, config) pairs
    std::map<HostPort, ConfigInfo> _configs;

    /// List of (hash, host:port) pairs (for looking up configs by hash)
    std::map<std::string, HostPort> _config_hashes;

    /// Combined hash of all configurations
    std::string _tracker_hash;

    mutable std::mutex _lock;
};

} // namespace kotekan

#endif // CONFIGTRACKER_H
