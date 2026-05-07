/**
 * @file
 * @brief Configuration tracker singleton and REST endpoints.
 *
 * Tracks the startup-time configuration JSON for participating kotekan instances.
 * The local config (this node's own startup config) is stored separately from
 * configs received from upstream peers so the local hash never depends on the
 * REST bind address (which a downstream peer may not see the same way).
 * Downstream peers fetch and re-key the upstream's local config under whatever
 * (host, port) they actually dialed, then propagate any further-upstream
 * configs the peer was already tracking.
 *
 * - kotekan::ConfigTracker
 *   -- instance
 *   -- n_configs
 *   -- getTrackerHash
 *   -- setLocalConfig
 *   -- insertUpstreamConfig
 *   -- hasLocalConfig
 *   -- hasUpstreamConfig
 *   -- getLocalConfigInfo
 *   -- getLocalConfigHash
 *   -- trackers_local_callback
 *   -- trackers_local_hash_callback
 *   -- trackers_upstream_configs_callback
 *   -- trackers_upstream_hashes_callback
 *   -- register_with_server
 *   -- getUpstreamConfigs
 *   -- getAllJSONConfigs
 *   -- writeConfigsToDisk
 *   -- reset
 */
#ifndef CONFIGTRACKER_H
#define CONFIGTRACKER_H

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO, ERROR_NON_OO, DEBUG_NON_OO, DEBUG2_NON_OO
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
#include <optional>      // for optional
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

public:
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

    /**
     * @brief Get the total number of configurations stored in the tracker
     * (local plus upstream).
     */
    std::size_t n_configs() const {
        std::lock_guard<std::mutex> lock(_lock);
        _check_upstream_consistent_locked();
        return (_local_config.has_value() ? 1u : 0u) + _upstream_configs.size();
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
     * @brief Set this node's local configuration.
     *
     * The local hash is computed over the JSON only (no host:port prefix), so
     * the local identity doesn't depend on which IP this node decides to
     * publish itself under. Downstream peers re-key it under whatever (host,
     * port) they actually dialed when they fetch it.
     *
     * Calling this with the same content as the existing local config is a
     * no-op. Calling it with different content is a fatal error: the local
     * config is set once at startup.
     *
     * Strips blocks containing a "kotekan_update_endpoint" before hashing.
     */
    void setLocalConfig(const nlohmann::json& config_json, const std::string& kotekan_version,
                        const std::string& kotekan_build_branch,
                        const std::string& kotekan_git_commit_hash,
                        const std::string& kotekan_cmake_options) {
        nlohmann::json filtered_json = _strip_update_endpoints(config_json);
        std::string json_hash = _jsonHashLocal(filtered_json);

        ConfigInfo info(filtered_json, json_hash, kotekan_version, kotekan_build_branch,
                        kotekan_git_commit_hash, kotekan_cmake_options);

        {
            std::lock_guard<std::mutex> lock(_lock);

            if (_local_config.has_value()) {
                if (!_config_info_matches(*_local_config, info)) {
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: setLocalConfig called with conflicting content "
                        "(existing hash: {}, new hash: {})",
                        _local_config->json_hash, info.json_hash);
                }
                return; // identical content; no-op
            }

            _local_config = info;
            _local_config_present_metric().labels({json_hash}).set(1.0);
            _refresh_count_metric_locked();
        }

        _setTrackerHash();
        DEBUG_NON_OO("ConfigTracker: set local config, hash: {}", json_hash);
    }

    /**
     * @brief Insert a config received from an upstream peer into the tracker.
     *
     * The upstream entry is keyed by (host, port). The hash bakes in the
     * (host, port) so that two distinct peers with identical configs still
     * produce distinct hashes (preserving the 1:1 invariant with
     * _upstream_config_hashes).
     *
     * @param host The (ipv4) host of the upstream peer, as observed by this
     *             node (i.e., the address this node dialed).
     * @param port The REST port of the upstream peer.
     * @param config_json The JSON configuration to insert.
     * @param kotekan_version The version of Kotekan.
     * @param kotekan_build_branch The build branch of Kotekan.
     * @param kotekan_git_commit_hash The git commit hash of Kotekan.
     * @param kotekan_cmake_options The CMake options used to build Kotekan.
     */
    void insertUpstreamConfig(std::string host, uint16_t port, const nlohmann::json& config_json,
                              const std::string& kotekan_version,
                              const std::string& kotekan_build_branch,
                              const std::string& kotekan_git_commit_hash,
                              const std::string& kotekan_cmake_options) {
        nlohmann::json filtered_json = _strip_update_endpoints(config_json);
        std::string json_hash = _jsonHashUpstream(filtered_json, host, port);

        ConfigInfo info(filtered_json, json_hash, kotekan_version, kotekan_build_branch,
                        kotekan_git_commit_hash, kotekan_cmake_options);

        _insertUpstreamConfig(host, port, info);
    }

    /**
     * @brief Check if a local config has been set.
     */
    bool hasLocalConfig() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config.has_value();
    }

    /**
     * @brief Check if an upstream config exists for the given (host, port).
     */
    bool hasUpstreamConfig(const std::string& host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _upstream_configs.count({host, port}) > 0;
    }

    /**
     * @brief Check if an upstream config with the given hash exists.
     */
    bool hasUpstreamConfig(const std::string& hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _upstream_config_hashes.count(hash) > 0;
    }

    /**
     * @brief Get the local ConfigInfo (a copy), or std::nullopt if not yet set.
     */
    std::optional<ConfigInfo> getLocalConfigInfo() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config;
    }

    /**
     * @brief Get the local config's hash, or "" if not yet set.
     */
    std::string getLocalConfigHash() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config.has_value() ? _local_config->json_hash : std::string{};
    }

    /**
     * @brief REST callback returning the local ConfigInfo as JSON, or 404 if
     * the local config hasn't been set yet.
     */
    void trackers_local_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        if (!_local_config.has_value()) {
            conn.send_error("local config not set", HTTP_RESPONSE::NOT_FOUND);
            return;
        }
        conn.send_json_reply(_local_config->to_json());
    }

    /**
     * @brief REST callback returning the local config's hash, or 404 if not
     * set yet. Cheaper than fetching the full local config.
     */
    void trackers_local_hash_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        if (!_local_config.has_value()) {
            conn.send_error("local config not set", HTTP_RESPONSE::NOT_FOUND);
            return;
        }
        nlohmann::json reply = {{"hash", _local_config->json_hash}};
        conn.send_json_reply(reply);
    }

    /**
     * @brief REST callback returning upstream configs.
     *
     * If the query string includes "hash=<...>", only the upstream config with
     * that hash is returned (or an empty object if no match).
     */
    void trackers_upstream_configs_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        _check_upstream_consistent_locked();

        nlohmann::json return_json = nlohmann::json::object();

        auto query_args = conn.get_query();
        std::string hash;
        if (query_args.find("hash") != query_args.end())
            hash = query_args["hash"];

        for (const auto& [host_port, info] : _upstream_configs) {
            if (!hash.empty() && info.json_hash != hash)
                continue;

            std::string host_port_str = host_port.host + ":" + std::to_string(host_port.port);
            return_json[host_port_str] = info.to_json();
        }

        conn.send_json_reply(return_json);
    }

    /**
     * @brief REST callback returning a JSON map of upstream-config hashes to
     * their (host, port).
     */
    void trackers_upstream_hashes_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        _check_upstream_consistent_locked();

        nlohmann::json return_json = nlohmann::json::object();
        for (const auto& [hash, host_port] : _upstream_config_hashes) {
            return_json[hash] = {{"host", host_port.host}, {"port", host_port.port}};
        }
        conn.send_json_reply(return_json);
    }

    /**
     * @brief Register the tracker's REST endpoints with the given server.
     *
     * Endpoints:
     *   GET /config_tracker_local            -> this node's local ConfigInfo
     *   GET /config_tracker_local_hash       -> {"hash": "..."} for the local config
     *   GET /config_tracker_upstream_configs -> all upstream configs (or the one matching ?hash=)
     *   GET /config_tracker_upstream_hashes  -> {hash: {host, port}} for upstream configs
     */
    void register_with_server(restServer* rest_server) {
        using namespace std::placeholders;
        rest_server->register_get_callback(
            "/config_tracker_local",
            std::bind(&ConfigTracker::trackers_local_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_local_hash",
            std::bind(&ConfigTracker::trackers_local_hash_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_upstream_configs",
            std::bind(&ConfigTracker::trackers_upstream_configs_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_upstream_hashes",
            std::bind(&ConfigTracker::trackers_upstream_hashes_callback, this, _1));
    }

    /**
     * @brief Fetch and insert an upstream peer's tracker state.
     *
     * Two-step protocol against the peer at (host, port):
     *   1. GET /config_tracker_local. The returned ConfigInfo is the peer's
     *      own local config; this node re-keys it under (host, port) (the
     *      address it actually dialed) and stores it as an upstream entry.
     *      The dialed (host, port) is the validated identity of the peer from
     *      this node's perspective, regardless of how the peer self-named.
     *   2. GET /config_tracker_upstream_hashes. For each hash not already
     *      present locally, GET /config_tracker_upstream_configs?hash=<...>
     *      and insert it as-is (its host:port reflects what the peer
     *      observed, which we trust transitively).
     *
     * Network/parse errors on individual fetches are logged and counted but
     * not fatal. Conflict between an existing entry and new content for the
     * same (host, port) is fatal.
     */
    void getUpstreamConfigs(const std::string& host, uint16_t port) {
        // Step 1: fetch and re-key the peer's local config.
        if (!_fetch_and_insert_peer_local(host, port)) {
            // Already logged; bail without attempting step 2 if we can't even
            // talk to the peer.
            return;
        }

        // Step 2: fetch the peer's upstream-hashes index.
        restClient::restReply reply = restClient::instance().make_request_blocking(
            "/config_tracker_upstream_hashes", {}, host, port, 1, -1);
        if (!reply.first) {
            ERROR_NON_OO("ConfigTracker: failed to GET /config_tracker_upstream_hashes from {}:{}",
                         host, port);
            _record_upstream_fetch(host, port, false);
            return;
        }

        nlohmann::json hashes_json;
        try {
            hashes_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            ERROR_NON_OO("ConfigTracker: failed to parse upstream-hashes response from {}:{}: {}",
                         host, port, e.what());
            DEBUG2_NON_OO("Response was: {}", reply.second);
            _record_upstream_fetch(host, port, false);
            return;
        }
        _record_upstream_fetch(host, port, true);

        for (const auto& item : hashes_json.items()) {
            const std::string& hash = item.key();
            const nlohmann::json& host_port_json = item.value();
            std::string upstream_host = host_port_json.value("host", host);
            uint16_t upstream_port = host_port_json.value("port", port);

            {
                std::lock_guard<std::mutex> lock(_lock);
                auto it = _upstream_config_hashes.find(hash);
                if (it != _upstream_config_hashes.end()) {
                    if (it->second.host == upstream_host && it->second.port == upstream_port)
                        continue;
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: hash {} already known for {}:{}, but peer {}:{} reports "
                        "it as {}:{}",
                        hash, it->second.host, it->second.port, host, port, upstream_host,
                        upstream_port);
                }
            }

            _fetch_and_insert_peer_upstream(host, port, upstream_host, upstream_port, hash);
        }

        // Sanity check
        std::lock_guard<std::mutex> lock(_lock);
        _check_upstream_consistent_locked();
    }

    /**
     * @brief Get a vector of strings of all config json (local + upstream).
     */
    std::vector<std::string> getAllJSONConfigs() const {
        std::vector<std::string> configs;
        std::lock_guard<std::mutex> lock(_lock);
        if (_local_config.has_value()) {
            configs.push_back(
                _local_config->to_json().dump(4, ' ', false,
                                              nlohmann::json::error_handler_t::strict));
        }
        for (const auto& [host_port, info] : _upstream_configs) {
            configs.push_back(
                info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        return configs;
    }

    /**
     * @brief Write all configuration data (local + upstream) to disk.
     *
     * The local config is written to "local.json" (no host:port available);
     * each upstream config is written to "<host>_<port>.json".
     *
     * @param directory The directory to write files to
     * @return Number of configurations successfully written
     */
    size_t writeConfigsToDisk(const std::string& directory) const {
        struct stat info;
        if (stat(directory.c_str(), &info) != 0) {
            FATAL_ERROR_NON_OO("ConfigTracker: error stating directory {}: {}", directory,
                               strerror(errno));
        }
        if (!S_ISDIR(info.st_mode)) {
            FATAL_ERROR_NON_OO("ConfigTracker: path is not a directory: {}", directory);
        }

        std::lock_guard<std::mutex> lock(_lock);
        size_t written = 0;

        if (_local_config.has_value()) {
            const std::string filename = directory + "/local.json";
            _write_one_atomic(filename, *_local_config);
            ++written;
        }

        for (const auto& [host_port, config_info] : _upstream_configs) {
            std::ostringstream filename_stream;
            filename_stream << directory << "/" << host_port.host << "_" << host_port.port
                            << ".json";
            _write_one_atomic(filename_stream.str(), config_info);
            ++written;
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
        std::vector<std::tuple<std::string, uint16_t, std::string>> cleared_upstream;
        std::string cleared_local_hash;
        {
            std::lock_guard<std::mutex> lock(_lock);
            for (const auto& [host_port, info] : _upstream_configs) {
                cleared_upstream.emplace_back(host_port.host, host_port.port, info.json_hash);
            }
            if (_local_config.has_value())
                cleared_local_hash = _local_config->json_hash;

            _upstream_configs.clear();
            _upstream_config_hashes.clear();
            _local_config.reset();
            _tracker_hash.clear();
            _configs_total_metric().set(0.0);
        }

        for (const auto& [host, port, hash] : cleared_upstream) {
            _config_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
        }
        if (!cleared_local_hash.empty())
            _local_config_present_metric().labels({cleared_local_hash}).set(0.0);

        _hash_changes_total().inc();
        _last_change_timestamp().set(_now_seconds());
    }

private:
    /// Constructor, we don't want anyone to call this
    ConfigTracker() = default;

    /// This node's own startup config (set once via setLocalConfig).
    /// Hash is over the JSON only; identity comes from the JSON content, not
    /// from any IP this node chose to publish.
    std::optional<ConfigInfo> _local_config;

    /// Configs received from upstream peers, keyed by the (host, port) this
    /// node observed for the peer. Hash includes (host, port).
    std::map<HostPort, ConfigInfo> _upstream_configs;

    /// Reverse lookup: upstream-config hash -> (host, port).
    std::map<std::string, HostPort> _upstream_config_hashes;

    /// Combined hash of all configurations (local first if present, then
    /// upstream hashes in sorted order).
    std::string _tracker_hash;

    mutable std::mutex _lock;

    static constexpr const char* _metrics_stage_name = "config_tracker";

    /**
     * @brief MD5 of a JSON config alone (no host:port).
     *
     * Used for the local config: the local hash is identity-by-content. The
     * caller must have stripped any kotekan_update_endpoint blocks.
     */
    std::string _jsonHashLocal(const nlohmann::json& filtered_json) const {
        if (_has_kotekan_update_endpoint(filtered_json)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _jsonHashLocal called with kotekan_update_endpoint present.");
        }
        std::string serialized =
            filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);
        return _md5_hex(serialized);
    }

    /**
     * @brief MD5 of (host, port, JSON config) so two distinct peers with
     * identical configs hash to distinct values. Caller must have stripped
     * any kotekan_update_endpoint blocks.
     */
    std::string _jsonHashUpstream(const nlohmann::json& filtered_json, const std::string& host,
                                  uint16_t port) const {
        if (_has_kotekan_update_endpoint(filtered_json)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _jsonHashUpstream called with kotekan_update_endpoint present.");
        }
        std::stringstream ss;
        ss << host << ":" << port << "|"
           << filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);
        return _md5_hex(ss.str());
    }

    /**
     * @brief Insert pre-built upstream ConfigInfo into the tracker.
     * Fatal if (host, port) already maps to different content, or if host/port
     * are invalid, or if the config still contains kotekan_update_endpoint.
     */
    void _insertUpstreamConfig(std::string host, uint16_t port, ConfigInfo config_info) {
        if (_has_kotekan_update_endpoint(config_info.config)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _insertUpstreamConfig called with kotekan_update_endpoint present.");
        }

        // normalize localhost to 127.0.0.1
        if (host == "localhost") {
            host = "127.0.0.1";
        }

        struct sockaddr_in sa4;
        if (inet_pton(AF_INET, host.c_str(), &(sa4.sin_addr)) != 1) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _insertUpstreamConfig called with invalid IPv4 address: {}", host);
        }
        if (port == 0) {
            FATAL_ERROR_NON_OO("ConfigTracker: _insertUpstreamConfig called with invalid port: {}",
                               port);
        }

        HostPort host_port{host, port};

        {
            std::lock_guard<std::mutex> lock(_lock);

            auto it = _upstream_configs.find(host_port);
            if (it != _upstream_configs.end()) {
                if (!_config_info_matches(it->second, config_info)) {
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: conflicting upstream configuration data present for "
                        "host: {}, port: {}",
                        host, port);
                }
                return; // identical content; no-op
            }

            _upstream_configs.emplace(host_port, config_info);
            _upstream_config_hashes.emplace(config_info.json_hash, host_port);
            _config_present_metric()
                .labels({host_port.host, std::to_string(host_port.port), config_info.json_hash})
                .set(1.0);
            _refresh_count_metric_locked();
        }

        _setTrackerHash();
        DEBUG_NON_OO("ConfigTracker: inserted upstream config for {}:{}, hash: {}", host, port,
                     config_info.json_hash);
    }

    /**
     * @brief Step 1 of getUpstreamConfigs: pull /config_tracker_local from
     * (host, port) and re-key it under that (host, port) as an upstream entry.
     *
     * Returns true on success (or "already had it"), false if the network
     * call failed.
     */
    bool _fetch_and_insert_peer_local(const std::string& host, uint16_t port) {
        restClient::restReply reply = restClient::instance().make_request_blocking(
            "/config_tracker_local", {}, host, port, 1, -1);
        if (!reply.first) {
            ERROR_NON_OO("ConfigTracker: failed to GET /config_tracker_local from {}:{}", host,
                         port);
            _record_upstream_fetch(host, port, false);
            return false;
        }

        nlohmann::json local_json;
        try {
            local_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            ERROR_NON_OO("ConfigTracker: failed to parse /config_tracker_local response from "
                         "{}:{}: {}",
                         host, port, e.what());
            DEBUG2_NON_OO("Response was: {}", reply.second);
            _record_upstream_fetch(host, port, false);
            return false;
        }

        ConfigInfo peer_local;
        try {
            peer_local = ConfigInfo(local_json);
        } catch (const std::exception& e) {
            ERROR_NON_OO("ConfigTracker: malformed /config_tracker_local payload from {}:{}: {}",
                         host, port, e.what());
            _record_upstream_fetch(host, port, false);
            return false;
        }

        // Re-key under the dialed (host, port) and re-hash accordingly.
        nlohmann::json filtered = _strip_update_endpoints(peer_local.config);
        peer_local.config = filtered;
        peer_local.json_hash = _jsonHashUpstream(filtered, host, port);

        _insertUpstreamConfig(host, port, peer_local);
        _record_upstream_fetch(host, port, true);
        return true;
    }

    /**
     * @brief Step 2 helper of getUpstreamConfigs: fetch a single upstream
     * config by hash from the peer at (peer_host, peer_port), validate it,
     * and insert under the (upstream_host, upstream_port) the peer reported.
     */
    void _fetch_and_insert_peer_upstream(const std::string& peer_host, uint16_t peer_port,
                                         const std::string& upstream_host, uint16_t upstream_port,
                                         const std::string& expected_hash) {
        const std::string path =
            std::string("/config_tracker_upstream_configs?hash=") + expected_hash;
        restClient::restReply reply =
            restClient::instance().make_request_blocking(path, {}, peer_host, peer_port, 1, -1);
        if (!reply.first) {
            ERROR_NON_OO(
                "ConfigTracker: failed to GET upstream config for hash {} from peer {}:{}",
                expected_hash, peer_host, peer_port);
            _record_upstream_fetch(upstream_host, upstream_port, false);
            return;
        }

        nlohmann::json response;
        try {
            response = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            ERROR_NON_OO(
                "ConfigTracker: failed to parse upstream-config response for hash {}: {}",
                expected_hash, e.what());
            _record_upstream_fetch(upstream_host, upstream_port, false);
            return;
        }

        const std::string host_port_str =
            upstream_host + ":" + std::to_string(upstream_port);
        if (!response.contains(host_port_str)) {
            ERROR_NON_OO("ConfigTracker: peer {}:{} did not return config for {} (hash {})",
                         peer_host, peer_port, host_port_str, expected_hash);
            _record_upstream_fetch(upstream_host, upstream_port, false);
            return;
        }

        ConfigInfo info;
        try {
            info = ConfigInfo(response[host_port_str]);
        } catch (const std::exception& e) {
            ERROR_NON_OO("ConfigTracker: malformed upstream-config payload for {} (hash {}): {}",
                         host_port_str, expected_hash, e.what());
            _record_upstream_fetch(upstream_host, upstream_port, false);
            return;
        }

        if (_jsonHashUpstream(info.config, upstream_host, upstream_port) != expected_hash
            || info.json_hash != expected_hash) {
            ERROR_NON_OO("ConfigTracker: hash mismatch for {} (expected {}); dropping",
                         host_port_str, expected_hash);
            _record_upstream_fetch(upstream_host, upstream_port, false);
            return;
        }

        _insertUpstreamConfig(upstream_host, upstream_port, info);
        _record_upstream_fetch(upstream_host, upstream_port, true);
    }

    /**
     * @brief Set a hash of the combined hash of all configurations stored in the tracker.
     * i.e. a hash representing the current tracker state.
     *
     * Local hash (if present) comes first, then upstream hashes in sorted
     * order (std::map iteration). Local-vs-upstream ordering is fixed so the
     * combined hash is deterministic regardless of insertion order.
     */
    void _setTrackerHash() {
        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(_lock);
            std::stringstream ss;
            if (_local_config.has_value())
                ss << _local_config->json_hash;
            for (const auto& [hash, _] : _upstream_config_hashes)
                ss << hash;

            const std::string new_hash = _md5_hex(ss.str());
            changed = (_tracker_hash != new_hash);
            _tracker_hash = new_hash;
            DEBUG_NON_OO("ConfigTracker: combined tracker hash is now {}", _tracker_hash);
        }
        if (changed) {
            _hash_changes_total().inc();
            _last_change_timestamp().set(_now_seconds());
        }
    }

    /// Compare two ConfigInfos for full equality (hash + version metadata).
    static bool _config_info_matches(const ConfigInfo& a, const ConfigInfo& b) {
        return a.json_hash == b.json_hash && a.kotekan_version == b.kotekan_version
               && a.kotekan_build_branch == b.kotekan_build_branch
               && a.kotekan_git_commit_hash == b.kotekan_git_commit_hash
               && a.kotekan_cmake_options == b.kotekan_cmake_options;
    }

    /// MD5 of a string, returned as a 32-char lowercase hex string.
    static std::string _md5_hex(const std::string& input) {
        unsigned char digest[MD5_DIGEST_LENGTH];

        // The MD5 function is deprecated in openssl 3.0, but we want to
        // maintain compatibility.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MD5(reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), digest);
#pragma GCC diagnostic pop

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
            ss << std::setw(2) << static_cast<int>(digest[i]);
        return ss.str();
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

    /// Atomically write a single ConfigInfo to disk by writing to a .tmp file
    /// and renaming. Fatal on any I/O error.
    void _write_one_atomic(const std::string& filename, const ConfigInfo& info) const {
        const std::string temp_filename = filename + ".tmp";
        try {
            {
                std::ofstream file(temp_filename);
                if (!file)
                    FATAL_ERROR_NON_OO("ConfigTracker: cannot open {} for writing", temp_filename);
                file << info.to_json().dump(4, ' ', false,
                                            nlohmann::json::error_handler_t::strict);
                if (!file.good())
                    FATAL_ERROR_NON_OO("ConfigTracker: write failed for {}", temp_filename);
            }
            if (std::rename(temp_filename.c_str(), filename.c_str()) != 0) {
                std::remove(temp_filename.c_str());
                FATAL_ERROR_NON_OO("ConfigTracker: failed to rename {} -> {}", temp_filename,
                                   filename);
            }
        } catch (const std::exception& e) {
            std::remove(temp_filename.c_str());
            FATAL_ERROR_NON_OO("ConfigTracker: error writing {}: {}", filename, e.what());
        }
    }

    /// Sanity check: _upstream_configs and _upstream_config_hashes must have
    /// the same size. Caller must hold _lock.
    void _check_upstream_consistent_locked() const {
        if (_upstream_configs.size() != _upstream_config_hashes.size()) {
            FATAL_ERROR_NON_OO("ConfigTracker: _upstream_configs ({}) and _upstream_config_hashes "
                               "({}) sizes differ",
                               _upstream_configs.size(), _upstream_config_hashes.size());
        }
    }

    /// Refresh the total-configs gauge. Caller must hold _lock.
    void _refresh_count_metric_locked() {
        _configs_total_metric().set(
            static_cast<double>((_local_config.has_value() ? 1u : 0u) + _upstream_configs.size()));
    }

    void _record_upstream_fetch(const std::string& host, uint16_t port, bool success) {
        _upstream_fetch_total()
            .labels({host, std::to_string(port), success ? "success" : "fail"})
            .inc();
    }

    double _now_seconds() const {
        using clock = std::chrono::steady_clock;
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
                                                      _metrics_stage_name,
                                                      {"host", "port", "hash"});
        return metric;
    }

    static prometheus::MetricFamily<prometheus::Gauge>& _local_config_present_metric() {
        static prometheus::MetricFamily<prometheus::Gauge>& metric =
            prometheus::Metrics::instance().add_gauge(
                "kotekan_config_tracker_local_config_present", _metrics_stage_name, {"hash"});
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
            prometheus::Metrics::instance().add_counter(
                "kotekan_config_tracker_upstream_fetch_total", _metrics_stage_name,
                {"host", "port", "result"});
        return metric;
    }

}; // class ConfigTracker

} // namespace kotekan

#endif // CONFIGTRACKER_H
