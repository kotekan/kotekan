/**
 * @file
 * @brief Configuration tracker singleton and REST endpoints.
 *
 * Tracks the startup-time configuration JSON for participating kotekan instances,
 * exposes REST endpoints to query stored configs and their hashes, maintains a combined
 * tracker hash, and supports persistence and synchronization with upstream trackers.
 *
 * - kotekan::ConfigTracker
 *   -- instance
 *   -- n_configs
 *   -- check_num_configs_consistent
 *   -- getTrackerHash
 *   -- insertRawConfig
 *   -- hasConfig
 *   -- trackers_configs_callback
 *   -- trackers_hashes_callback
 *   -- register_with_server
 *   -- getUpstreamConfigs
 *   -- writeConfigsToDisk
 *   -- reset
 */
#ifndef CONFIGTRACKER_H
#define CONFIGTRACKER_H

#include "kotekanLogging.hpp"    // for FATAL_ERROR_NON_OO, ERROR_NON_OO, DEBUG_NON_OO, DEBUG2_NON_OO
#include "prometheusMetrics.hpp" // for Counter, Gauge, MetricFamily, Metrics
#include "restClient.hpp"        // for restClient
#include "restServer.hpp"        // for connectionInstance, restServer

#include "fmt.hpp"  // for compile_string_to_view
#include "json.hpp" // for iter_impl, json, json_ref, iteration_proxy_value, basic_json

#include <arpa/inet.h>   // for inet_pton
#include <chrono>        // for duration, duration_cast, system_clock
#include <cstdio>        // for remove, rename, size_t
#include <errno.h>       // for errno
#include <exception>     // for exception
#include <fstream>       // for basic_ofstream
#include <functional>    // for bind, _1, function
#include <iomanip>       // for operator<<, setfill, setw
#include <map>           // for map, operator!=, _Rb_tree_iterator, _Rb_tree_const_iterator
#include <mutex>         // for mutex, lock_guard
#include <netinet/in.h>  // for sockaddr_in
#include <openssl/md5.h> // for MD5, MD5_DIGEST_LENGTH
#include <sstream>       // for basic_ostream, basic_stringstream, operator<<, basic_ostre...
#include <stdint.h>      // for uint16_t
#include <string.h>      // for strerror
#include <string>        // for basic_string, allocator, char_traits, operator+, string
#include <sys/socket.h>  // for AF_INET
#include <sys/stat.h>    // for stat, S_ISDIR
#include <tuple>         // for tie, operator<, tuple
#include <utility>       // for pair
#include <vector>        // for vector

namespace kotekan {

/**
 * @class ConfigTracker
 * @brief Kotekan core component that tracks the (startup-time) configurations through a pipeline.
 *
 * The rest callbacks must be registered with a kotekan REST server instance by
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

    ~ConfigTracker() {}

private:
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
        nlohmann::json
            config; /// Configuration data json (minus blocks with kotekan_update_endpoint)
        std::string json_hash; /// Stored md5 hash of the ConfigInfo::config

        /// Kotekan version information (should match lib/version details.)
        std::string kotekan_version;
        /// Kotekan git branch (should match lib/version details.)
        std::string kotekan_build_branch;
        /// Kotekan git commit (should match lib/version details.)
        std::string kotekan_git_commit_hash;
        /// Kotekan build options information (should match lib/version details.)
        std::string kotekan_cmake_options;

        // Default constructor
        ConfigInfo() = default;

        // Constructor with all parameters
        ConfigInfo(const nlohmann::json& config, const std::string& json_hash,
                   const std::string& kotekan_version, const std::string& kotekan_build_branch,
                   const std::string& kotekan_git_commit_hash,
                   const std::string& kotekan_cmake_options) :
            config(config), json_hash(json_hash), kotekan_version(kotekan_version),
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

        // "from_json" function for nlohmann::json
        static ConfigInfo from_json(const nlohmann::json& j) {
            return ConfigInfo(j);
        }

        // Convert to JSON
        nlohmann::json to_json() const {
            return nlohmann::json{{"config", config},
                                  {"json_hash", json_hash},
                                  {"kotekan_version", kotekan_version},
                                  {"kotekan_build_branch", kotekan_build_branch},
                                  {"kotekan_git_commit_hash", kotekan_git_commit_hash},
                                  {"kotekan_cmake_options", kotekan_cmake_options}};
        }
    };

public:
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
     * @brief Check if the number of configs and hashes are consistent.
     * (Mainly useful for debugging, should always be true. Exposed for boost tests.)
     *
     * This function checks if the number of configurations stored in _configs
     * matches the number of hashes stored in _config_hashes.
     *
     * @returns True if the sizes are consistent, false otherwise.
     */
    void check_num_configs_consistent() const {
        std::lock_guard<std::mutex> lock(_lock);
        // _configs and _config_hashes should always have the same size

        if (_configs.size() != _config_hashes.size()) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _configs and _config_hashes have different sizes: {} vs {}",
                _configs.size(), _config_hashes.size());
        }
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
     * @brief Insert raw (unfiltered) JSON configuration into the tracker.
     * Creates a ConfigInfo object from the parameters.
     * This function will strip blocks with an kotekan_update_endpoint.
     *
     * @param host The host where the kotekan instance with the configuration is running.
     * @param port The port where the kotekan instance with the configuration is running.
     * @param config_json The JSON configuration to insert.
     * @param kotekan_version The version of Kotekan.
     * @param kotekan_build_branch The build branch of Kotekan.
     * @param kotekan_git_commit_hash The git commit hash of Kotekan.
     * @param kotekan_cmake_options The CMake options used to build Kotekan.
     */
    void insertRawConfig(std::string host, uint16_t port, const nlohmann::json& config_json,
                         const std::string& kotekan_version,
                         const std::string& kotekan_build_branch,
                         const std::string& kotekan_git_commit_hash,
                         const std::string& kotekan_cmake_options) {

        // Strip blocks with an kotekan_update_endpoint before hashing
        nlohmann::json filtered_json = _strip_update_endpoints(config_json);
        std::string json_hash = _jsonHash(filtered_json);

        ConfigInfo info =
            ConfigInfo(filtered_json, json_hash, kotekan_version, kotekan_build_branch,
                       kotekan_git_commit_hash, kotekan_cmake_options);

        // Call _insertConfig with the constructed ConfigInfo
        _insertConfig(host, port, info);
    }

    /**
     * @brief Check if a config exists in the _configs map.
     *
     * @param hash The hash of the config.
     * @returns True if the config exists, false otherwise.
     */
    bool hasConfig(std::string host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _configs.count({host, port}) > 0;
    }

    /**
     * @brief Check if a config exists using a hash string instead of host and port.
     */
    bool hasConfig(std::string hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _config_hashes.count(hash) > 0;
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
        // If a hash is provided, only return the config with that hash
        std::string hash = "";
        if (query_args.find("hash") != query_args.end()) {
            hash = query_args["hash"];
        }

        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& config : _configs) {
            if (!hash.empty() && config.second.json_hash != hash)
                continue;

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
     * @brief Registers this object with the REST server, creating the
     *        /config_tracker_configs and config_tracker_hashes end points.
     * If a hash is provided to /config_tracker_configs, it will return the config for that hash.
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server) {
        using namespace std::placeholders;
        // register callback for /config_tracker_*, pass along hash if provided
        rest_server->register_get_callback(
            "/config_tracker_configs",
            std::bind(&ConfigTracker::trackers_configs_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_hashes",
            std::bind(&ConfigTracker::trackers_hashes_callback, this, _1));
    }

    /**
     * @brief Request and insert upstream configs into the tracker.
     * This function will fetch hashes from an upstream REST server, then:
     *   1) check if upstream configs are already present locally, if not, fetch them.
     *   2) If the hash already exists, check to make sure the (hash, host:port) hasn't changed
     * (otherwise fail).
     */
    void getUpstreamConfigs(const std::string& host, uint16_t port) {

        // Send a request to the upstream server to get all config hashes.
        nlohmann::json request_json = {};
        restClient::restReply reply = restClient::instance().make_request_blocking(
            "/config_tracker_hashes", {}, host, port, 1, -1);
        // reply is a pair with success boolean and the reply string
        if (!reply.first) {
            ERROR_NON_OO(
                "ConfigTracker: Failed to get config hashes from upstream host: {}, port: {}", host,
                port);
            _record_upstream_fetch(host, port, false);
            return;
        }

        // Check if the response contains any hashes
        if (reply.second.empty()) {
            ERROR_NON_OO("ConfigTracker: No configs found at upstream host: {}, port: {}", host,
                         port);
            _record_upstream_fetch(host, port, false);
            return;
        }

        // Convert the reply string to a JSON object
        nlohmann::json response_json;
        try {
            response_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            // Error, but not fatal if we can't parse the JSON
            ERROR_NON_OO("ConfigTracker: Failed to parse JSON response from upstream host: {}, "
                         "port: {}. Error: {}",
                         host, port, e.what());
            DEBUG2_NON_OO("Response was: {}", reply.second);
            _record_upstream_fetch(host, port, false);
            return;
        }
        _record_upstream_fetch(host, port, true);

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
                if (_config_hashes[hash].host == upstream_host
                    && _config_hashes[hash].port == upstream_port) {
                    // Config already exists with the same host and port, continue
                    continue;
                } else {
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: Hash conflict for {}, upstream host: {}, port: {}", hash,
                        upstream_host, upstream_port);
                }
            }

            // If it doesn't exist, fetch the config from the immediate upstream server.
            // Use GET with query string to match the registered GET endpoint.
            const std::string path_configs = std::string("/config_tracker_configs?hash=") + hash;
            reply =
                restClient::instance().make_request_blocking(path_configs, {}, host, port, 1, -1);
            // Check if the request was successful
            if (!reply.first) {
                ERROR_NON_OO("ConfigTracker: Failed to get config for hash: {} from upstream host: "
                             "{}, port: {}",
                             hash, upstream_host, upstream_port);
                DEBUG2_NON_OO("Response was: {}", reply.second);
                _record_upstream_fetch(upstream_host, upstream_port, false);
                continue;
            }
            // Parse the response JSON
            nlohmann::json config_response_json;
            try {
                config_response_json = nlohmann::json::parse(reply.second);
            } catch (const nlohmann::json::parse_error& e) {
                ERROR_NON_OO("ConfigTracker: Failed to parse JSON response for hash: {} from "
                             "upstream host: {}, port: {}. Error: {}",
                             hash, upstream_host, upstream_port, e.what());
                _record_upstream_fetch(upstream_host, upstream_port, false);
                continue;
            }

            // Check if the response contains the config under the owner
            // (upstream_host:upstream_port)
            std::string host_port_str = upstream_host + ":" + std::to_string(upstream_port);
            if (config_response_json.contains(host_port_str)) {
                ConfigInfo info = ConfigInfo(config_response_json[host_port_str]);
                // check that the new hash matches expectations.
                if (_jsonHash(info.config) != hash || info.json_hash != hash)
                    ERROR_NON_OO(
                        "ConfigTracker: Returned hash or config is inconsistent with hash {}!",
                        hash);
                _insertConfig(upstream_host, upstream_port, info);
                _record_upstream_fetch(upstream_host, upstream_port, true);
            } else {
                // If the config was not found, log a non-fatal error.
                ERROR_NON_OO("ConfigTracker: Config not found for hash: {}", hash);
                _record_upstream_fetch(upstream_host, upstream_port, false);
            }
        }
        // Sanity check to ensure that the number of configs is consistent
        check_num_configs_consistent();
    }

    /**
     * @brief Get a vector of strings of all config json
     */
    std::vector<std::string> getAllJSONConfigs() const {
        std::vector<std::string> configs;
        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& [host_port, config_info] : _configs) {
            configs.push_back(
                config_info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        return configs;
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
            std::string err =
                "Error stating directory: " + directory + ", error: " + strerror(errno);
            FATAL_ERROR_NON_OO("ConfigTracker: {}", err);
        }

        if (!S_ISDIR(info.st_mode)) {
            std::string err = "Path is not a directory: " + directory;
            FATAL_ERROR_NON_OO("ConfigTracker: {}", err);
        }

        std::lock_guard<std::mutex> lock(_lock);
        size_t written = 0;

        for (const auto& [host_port, config_info] : _configs) {
            // Create filename - use underscore instead of colon for portability
            std::ostringstream filename_stream;
            filename_stream << directory << "/" << host_port.host << "_" << host_port.port
                            << ".json";
            std::string filename = filename_stream.str();

            try {
                // Write atomically by writing to temp file first
                std::string temp_filename = filename + ".tmp";

                {
                    std::ofstream file(temp_filename);
                    if (!file) {
                        FATAL_ERROR_NON_OO("ConfigTracker: Cannot open file for writing");
                    }

                    // Write with pretty formatting
                    file << config_info.to_json().dump(4, ' ', false,
                                                       nlohmann::json::error_handler_t::strict);

                    if (!file.good()) {
                        FATAL_ERROR_NON_OO("ConfigTracker: Write failed");
                    }
                } // file closed

                // Atomically rename temp file to final name
                if (std::rename(temp_filename.c_str(), filename.c_str()) != 0) {
                    FATAL_ERROR_NON_OO("ConfigTracker: Failed to rename temp file");
                }

                ++written;

            } catch (const std::exception& e) {
                std::string err = "Error writing " + filename + ": " + e.what();
                FATAL_ERROR_NON_OO("{}", err);

                // Clean up temp file if it exists
                std::remove((filename + ".tmp").c_str());
            }
        }

        return written;
    }

    /**
     * @brief Clear all tracked configurations and hashes.
     *
     * Empties the internal maps and clears the combined tracker hash. Intended
     * primarily for tests or controlled re-initialization.
     */
    void reset() {
        std::vector<std::tuple<std::string, uint16_t, std::string>> cleared_configs;
        {
            std::lock_guard<std::mutex> lock(_lock);
            for (const auto& [host_port, info] : _configs) {
                cleared_configs.emplace_back(host_port.host, host_port.port, info.json_hash);
            }
            _configs.clear();
            _config_hashes.clear();
            _tracker_hash.clear();
            _configs_total_metric().set(0.0);
        }

        for (const auto& [host, port, hash] : cleared_configs) {
            _config_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
        }

        _hash_changes_total().inc();
        _last_change_timestamp().set(_now_seconds());
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

    static constexpr const char* _metrics_stage_name = "config_tracker";

    /**
     * @brief Get the md5 hash of a (json) config.
     *
     * This function generates a hash for a given configuration JSON object.
     * In the context of the configTracker, the JSON object must have any
     * blocks containing a "kotekan_update_endpoint" stripped before hashing.
     * (This function only checks for that, and errors, expecting a caller to
     * supply a compliant config json.)
     *
     * @param filtered_json The configuration JSON object to hash.
     * @returns The canonical hash as a string.
     */
    std::string _jsonHash(const nlohmann::json& filtered_json) const {
        std::stringstream ss;

        // nlohmann::json::dump() uses an alpha-ordered map for objects, so the
        // config gets serialized in a consistent order.

        // In order for this to hash configs correctly, this assumes any kotekan_update_endpoints
        // are removed, and versioning information has been added.
        if (_has_kotekan_update_endpoint(filtered_json)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _jsonHash called with kotekan_update_endpoint present.");
        }

        // Stick to a string dump for now
        // TODO: a binary dump storing the full double precision bit pattern could be more
        // consistent.
        ss << filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);

        std::string serialized = ss.str();
        unsigned char md5_result[MD5_DIGEST_LENGTH];

        // The MD5 function is deprecated in openssl 3.0, but we want to
        // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MD5(reinterpret_cast<const unsigned char*>(serialized.c_str()), serialized.size(),
            md5_result);
#pragma GCC diagnostic pop

        std::stringstream md5_ss;
        md5_ss << std::hex << std::setw(2) << std::setfill('0');
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
            md5_ss << static_cast<int>(md5_result[i]);

        return md5_ss.str();
    }

    /**
     * @brief Insert configuration information into the tracker.
     * Fatal kotekan error if a config with the same host and port already exists, are invalid, or
     * the config includes a kotekan_update_endpoint.
     *
     * @param host The (ipv4) host where the kotekan instance with the configuration is running.
     * @param port The port where the kotekan instance with the configuration is running.
     * @param config_info The configuration information to insert.
     */
    void _insertConfig(std::string host, uint16_t port, ConfigInfo config_info) {

        // Make sure the config_info doesn't have a "kotekan_update_endpoint" in its config
        if (_has_kotekan_update_endpoint(config_info.config)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _insertConfig called with kotekan_update_endpoint present.");
        }

        // normalize localhost to 127.0.0.1
        if (host == "localhost") {
            host = "127.0.0.1";
        }

        // Validate the host is a valid IPv4 address
        struct sockaddr_in sa4;
        if (inet_pton(AF_INET, host.c_str(), &(sa4.sin_addr)) != 1) {
            FATAL_ERROR_NON_OO("ConfigTracker: _insertConfig called with invalid IPv4 address: {}",
                               host);
        }

        // Validate the port is in a valid range. port < 65535 by definition of uint16_t.
        if (port == 0) {
            FATAL_ERROR_NON_OO("ConfigTracker: _insertConfig called with invalid port: {}", port);
        }

        HostPort host_port{host, port};

        // Store the config in the map with its hash
        {
            std::lock_guard<std::mutex> lock(_lock);

            if (_configs.count(host_port)) {
                // If a config already exists with the same host and port,
                // make sure the hash and version metadata match.
                if (_configs[host_port].json_hash != config_info.json_hash
                    || _configs[host_port].kotekan_version != config_info.kotekan_version
                    || _configs[host_port].kotekan_build_branch != config_info.kotekan_build_branch
                    || _configs[host_port].kotekan_git_commit_hash
                           != config_info.kotekan_git_commit_hash
                    || _configs[host_port].kotekan_cmake_options
                           != config_info.kotekan_cmake_options) {
                    FATAL_ERROR_NON_OO("ConfigTracker: conflicting configuration data present for "
                                       "host: {}, port: {}",
                                       host, port);
                }
                // Don't add anything
                return;
            }

            _configs.emplace(host_port, config_info);
            _config_hashes.emplace(config_info.json_hash, host_port);
            _refresh_metrics_locked();
        }
        // Sanity check to ensure that the number of configs is consistent
        check_num_configs_consistent();

        // Update the combined hash
        _setTrackerHash();
        {
            // lock again to print debug info
            std::lock_guard<std::mutex> lock(_lock);
            DEBUG_NON_OO("ConfigTracker: inserted config for host: {}, port: {}, hash: {}", host,
                         port, config_info.json_hash);
            DEBUG_NON_OO("ConfigTracker: _tracker_hash: {}", _tracker_hash);
        }
    }

    /**
     * @brief Set a hash of the combined hash of all configurations stored in the tracker.
     * i.e. a hash representing the current tracker state.
     */
    void _setTrackerHash() {

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
        md5_ss << std::hex << std::setw(2) << std::setfill('0');
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
            md5_ss << static_cast<int>(md5_result[i]);

        const std::string new_hash = md5_ss.str();
        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(_lock);
            changed = (_tracker_hash != new_hash);
            // Store the combined hash
            _tracker_hash = new_hash;
            DEBUG_NON_OO("ConfigTracker: Combined hash set to: {}", _tracker_hash);
        }
        if (changed) {
            _hash_changes_total().inc();
            _last_change_timestamp().set(_now_seconds());
        }
    }

    nlohmann::json _strip_update_endpoints(const nlohmann::json& j) const {
        if (j.is_object()) {
            if (j.contains("kotekan_update_endpoint"))
                return nullptr;

            nlohmann::json result = nlohmann::json::object();
            for (auto& [k, v] : j.items()) {
                if (auto sub = _strip_update_endpoints(v); !sub.is_null())
                    result[k] = sub;
            }
            return result;
        }
        if (j.is_array()) {
            nlohmann::json result = nlohmann::json::array();
            for (auto& v : j)
                if (auto sub = _strip_update_endpoints(v); !sub.is_null())
                    result.push_back(sub);
            return result;
        }
        return j;
    }

    bool _has_kotekan_update_endpoint(const nlohmann::json& j) const {
        if (j.is_object()) {
            if (j.contains("kotekan_update_endpoint"))
                return true;
            for (auto& [k, v] : j.items())
                if (_has_kotekan_update_endpoint(v))
                    return true;
        } else if (j.is_array()) {
            for (auto& v : j)
                if (_has_kotekan_update_endpoint(v))
                    return true;
        }
        return false;
    }

    void _refresh_metrics_locked() {
        _configs_total_metric().set(static_cast<double>(_configs.size()));
        for (const auto& [host_port, info] : _configs) {
            _config_present_metric()
                .labels({host_port.host, std::to_string(host_port.port), info.json_hash})
                .set(1.0);
        }
    }

    void _record_upstream_fetch(const std::string& host, uint16_t port, bool success) {
        _upstream_fetch_total()
            .labels({host, std::to_string(port), success ? "success" : "fail"})
            .inc();
    }

    double _now_seconds() const {
        using clock = std::chrono::system_clock;
        using seconds_double = std::chrono::duration<double>;
        return std::chrono::duration_cast<seconds_double>(clock::now().time_since_epoch()).count();
    }

    static prometheus::Gauge& _configs_total_metric() {
        static prometheus::Gauge& metric = prometheus::Metrics::instance().add_gauge(
            "kotekan_config_tracker_configs_total", _metrics_stage_name);
        return metric;
    }

    static prometheus::MetricFamily<prometheus::Gauge>& _config_present_metric() {
        static prometheus::MetricFamily<prometheus::Gauge>& metric =
            prometheus::Metrics::instance().add_gauge("kotekan_config_tracker_config_present",
                                                      _metrics_stage_name, {"host", "port", "hash"});
        return metric;
    }

    static prometheus::Counter& _hash_changes_total() {
        static prometheus::Counter& metric = prometheus::Metrics::instance().add_counter(
            "kotekan_config_tracker_hash_changes_total", _metrics_stage_name);
        return metric;
    }

    static prometheus::Gauge& _last_change_timestamp() {
        static prometheus::Gauge& metric = prometheus::Metrics::instance().add_gauge(
            "kotekan_config_tracker_last_change_timestamp_seconds", _metrics_stage_name);
        return metric;
    }

    static prometheus::MetricFamily<prometheus::Counter>& _upstream_fetch_total() {
        static prometheus::MetricFamily<prometheus::Counter>& metric =
            prometheus::Metrics::instance().add_counter("kotekan_config_tracker_upstream_fetch_total",
                                                        _metrics_stage_name,
                                                        {"host", "port", "result"});
        return metric;
    }

}; // class ConfigTracker

} // namespace kotekan

#endif // CONFIGTRACKER_H
