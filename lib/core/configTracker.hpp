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
 * In addition to peer kotekan configs, the tracker stores configuration and
 * timing data fetched from an upstream FPGA controller (typically by
 * dpdkCore). Those entries propagate to downstream kotekan nodes via REST so
 * that an HDF5 writer downstream sees them too. Any change to FPGA config or
 * FPGA timing after the initial fetch indicates a controller reset and is
 * fatal.
 *
 * - kotekan::ConfigTracker
 *   -- instance
 *   -- n_configs
 *   -- getTrackerHash
 *   -- setLocalConfig
 *   -- insertUpstreamConfig
 *   -- setFpgaConfig / setFpgaTiming
 *   -- insertFpgaConfig / insertFpgaTiming
 *   -- hasLocalConfig / hasUpstreamConfig / hasFpgaConfig / hasFpgaTiming
 *   -- getLocalConfigInfo / getLocalConfigHash
 *   -- trackers_local_callback
 *   -- trackers_local_hash_callback
 *   -- trackers_upstream_configs_callback / trackers_upstream_hashes_callback
 *   -- trackers_fpga_configs_callback / trackers_fpga_config_hashes_callback
 *   -- trackers_fpga_timings_callback / trackers_fpga_timing_hashes_callback
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
     *
     * For FPGA entries (config or timing) the version metadata fields are
     * unused and stored as empty strings.
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
     * (local + upstream + FPGA configs + FPGA timings).
     */
    std::size_t n_configs() const {
        std::lock_guard<std::mutex> lock(_lock);
        _check_consistent_locked(_upstream_configs, _upstream_config_hashes, "upstream");
        _check_consistent_locked(_fpga_configs, _fpga_config_hashes, "fpga config");
        _check_consistent_locked(_fpga_timings, _fpga_timing_hashes, "fpga timing");
        return (_local_config.has_value() ? 1u : 0u) + _upstream_configs.size()
               + _fpga_configs.size() + _fpga_timings.size();
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

        bool inserted = false;
        bool hash_changed = false;
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
            // Recompute the combined tracker hash in the same critical section
            // as the map update so readers never observe new entries with a
            // stale tracker hash.
            hash_changed = _recomputeTrackerHashLocked();
            inserted = true;
        }

        if (inserted) {
            if (hash_changed) {
                _hash_changes_total().inc();
                _last_change_timestamp().set(_now_seconds());
            }
            DEBUG_NON_OO("ConfigTracker: set local config, hash: {}", json_hash);
        }
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
        std::string json_hash = _jsonHashWithEndpoint(filtered_json, host, port);

        ConfigInfo info(filtered_json, json_hash, kotekan_version, kotekan_build_branch,
                        kotekan_git_commit_hash, kotekan_cmake_options);

        _insertCategorized(_upstream_configs, _upstream_config_hashes, host, port, info, "upstream",
                           _config_present_metric());
    }

    /**
     * @brief Store the upstream FPGA controller's startup configuration.
     *
     * The kotekan_* version metadata on FPGA entries is unused (stored as
     * empty strings). The host:port identifies the FPGA controller.
     *
     * Idempotent on identical content; FATAL on conflict (a content change
     * implies the controller reset).
     */
    void setFpgaConfig(std::string host, uint16_t port, const nlohmann::json& config_json) {
        ConfigInfo info = _build_fpga_info(host, port, config_json);
        _insertCategorized(_fpga_configs, _fpga_config_hashes, host, port, info, "fpga config",
                           _fpga_config_present_metric());
    }

    /**
     * @brief Store the upstream FPGA controller's timing snapshot.
     *
     * Stored alongside the FPGA config; treated identically (idempotent on
     * identical content; FATAL on conflict, indicating controller reset).
     */
    void setFpgaTiming(std::string host, uint16_t port, const nlohmann::json& timing_json) {
        ConfigInfo info = _build_fpga_info(host, port, timing_json);
        _insertCategorized(_fpga_timings, _fpga_timing_hashes, host, port, info, "fpga timing",
                           _fpga_timing_present_metric());
    }

    /**
     * @brief Insert a pre-built FPGA-config ConfigInfo (used by the
     * propagation path in getUpstreamConfigs).
     */
    void insertFpgaConfig(std::string host, uint16_t port, ConfigInfo info) {
        _insertCategorized(_fpga_configs, _fpga_config_hashes, host, port, std::move(info),
                           "fpga config", _fpga_config_present_metric());
    }

    /**
     * @brief Insert a pre-built FPGA-timing ConfigInfo (used by the
     * propagation path in getUpstreamConfigs).
     */
    void insertFpgaTiming(std::string host, uint16_t port, ConfigInfo info) {
        _insertCategorized(_fpga_timings, _fpga_timing_hashes, host, port, std::move(info),
                           "fpga timing", _fpga_timing_present_metric());
    }

    /// True if a local config has been set.
    bool hasLocalConfig() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config.has_value();
    }

    /// True if an upstream entry exists for the given (host, port).
    bool hasUpstreamConfig(const std::string& host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _upstream_configs.count({host, port}) > 0;
    }

    /// True if an upstream entry exists with the given hash.
    bool hasUpstreamConfig(const std::string& hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _upstream_config_hashes.count(hash) > 0;
    }

    /// True if an FPGA config entry exists for the given controller (host, port).
    bool hasFpgaConfig(const std::string& host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _fpga_configs.count({host, port}) > 0;
    }

    /// True if an FPGA config entry exists with the given hash.
    bool hasFpgaConfig(const std::string& hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _fpga_config_hashes.count(hash) > 0;
    }

    /// True if an FPGA timing entry exists for the given controller (host, port).
    bool hasFpgaTiming(const std::string& host, uint16_t port) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _fpga_timings.count({host, port}) > 0;
    }

    /// True if an FPGA timing entry exists with the given hash.
    bool hasFpgaTiming(const std::string& hash) const {
        std::lock_guard<std::mutex> lock(_lock);
        return _fpga_timing_hashes.count(hash) > 0;
    }

    /// Copy of the local ConfigInfo, or std::nullopt if not yet set.
    std::optional<ConfigInfo> getLocalConfigInfo() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config;
    }

    /// Local config's hash, or "" if not yet set.
    std::string getLocalConfigHash() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _local_config.has_value() ? _local_config->json_hash : std::string{};
    }

    /// REST callback returning the local ConfigInfo as JSON, or 404.
    void trackers_local_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        if (!_local_config.has_value()) {
            conn.send_error("local config not set", HTTP_RESPONSE::NOT_FOUND);
            return;
        }
        conn.send_json_reply(_local_config->to_json());
    }

    /// REST callback returning {"hash": "..."} for the local config, or 404.
    void trackers_local_hash_callback(connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(_lock);
        if (!_local_config.has_value()) {
            conn.send_error("local config not set", HTTP_RESPONSE::NOT_FOUND);
            return;
        }
        nlohmann::json reply = {{"hash", _local_config->json_hash}};
        conn.send_json_reply(reply);
    }

    /// REST callback returning upstream-peer configs (optionally filtered by ?hash=).
    void trackers_upstream_configs_callback(connectionInstance& conn) {
        _serve_categorized_entries(conn, _upstream_configs, _upstream_config_hashes, "upstream");
    }

    /// REST callback returning upstream-peer hash -> {host, port}.
    void trackers_upstream_hashes_callback(connectionInstance& conn) {
        _serve_categorized_hashes(conn, _upstream_configs, _upstream_config_hashes, "upstream");
    }

    /// REST callback returning FPGA configs (optionally filtered by ?hash=).
    void trackers_fpga_configs_callback(connectionInstance& conn) {
        _serve_categorized_entries(conn, _fpga_configs, _fpga_config_hashes, "fpga config");
    }

    /// REST callback returning FPGA-config hash -> {host, port}.
    void trackers_fpga_config_hashes_callback(connectionInstance& conn) {
        _serve_categorized_hashes(conn, _fpga_configs, _fpga_config_hashes, "fpga config");
    }

    /// REST callback returning FPGA timings (optionally filtered by ?hash=).
    void trackers_fpga_timings_callback(connectionInstance& conn) {
        _serve_categorized_entries(conn, _fpga_timings, _fpga_timing_hashes, "fpga timing");
    }

    /// REST callback returning FPGA-timing hash -> {host, port}.
    void trackers_fpga_timing_hashes_callback(connectionInstance& conn) {
        _serve_categorized_hashes(conn, _fpga_timings, _fpga_timing_hashes, "fpga timing");
    }

    /**
     * @brief Register the tracker's REST endpoints with the given server.
     *
     * Endpoints:
     *   GET /config_tracker_local                  -> this node's local ConfigInfo
     *   GET /config_tracker_local_hash             -> {"hash": "..."} for the local config
     *   GET /config_tracker_upstream_configs       -> upstream peer configs (or ?hash=)
     *   GET /config_tracker_upstream_hashes        -> {hash: {host, port}} for upstream
     *   GET /config_tracker_fpga_configs           -> FPGA configs (or ?hash=)
     *   GET /config_tracker_fpga_config_hashes     -> {hash: {host, port}} for FPGA configs
     *   GET /config_tracker_fpga_timings           -> FPGA timings (or ?hash=)
     *   GET /config_tracker_fpga_timing_hashes     -> {hash: {host, port}} for FPGA timings
     */
    void register_with_server(restServer* rest_server) {
        using namespace std::placeholders;
        rest_server->register_get_callback(
            "/config_tracker_local", std::bind(&ConfigTracker::trackers_local_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_local_hash",
            std::bind(&ConfigTracker::trackers_local_hash_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_upstream_configs",
            std::bind(&ConfigTracker::trackers_upstream_configs_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_upstream_hashes",
            std::bind(&ConfigTracker::trackers_upstream_hashes_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_fpga_configs",
            std::bind(&ConfigTracker::trackers_fpga_configs_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_fpga_config_hashes",
            std::bind(&ConfigTracker::trackers_fpga_config_hashes_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_fpga_timings",
            std::bind(&ConfigTracker::trackers_fpga_timings_callback, this, _1));
        rest_server->register_get_callback(
            "/config_tracker_fpga_timing_hashes",
            std::bind(&ConfigTracker::trackers_fpga_timing_hashes_callback, this, _1));
    }

    /**
     * @brief Fetch and insert an upstream peer's tracker state.
     *
     * Protocol against the peer at (host, port):
     *   1. GET /config_tracker_local. The returned ConfigInfo is the peer's
     *      own local config; this node re-keys it under (host, port) (the
     *      address it actually dialed) and stores it as an upstream entry.
     *   2. GET /config_tracker_upstream_hashes. For each hash not already
     *      present locally, GET /config_tracker_upstream_configs?hash=<...>
     *      and insert it (its host:port reflects what the peer observed,
     *      which we trust transitively).
     *   3. GET /config_tracker_fpga_config_hashes; for each missing hash, GET
     *      /config_tracker_fpga_configs?hash=<...> and insert.
     *   4. GET /config_tracker_fpga_timing_hashes; for each missing hash, GET
     *      /config_tracker_fpga_timings?hash=<...> and insert.
     *
     * Network/parse errors on individual fetches are logged and counted but
     * not fatal. Conflict between an existing entry and new content for the
     * same (host, port) is fatal.
     */
    void getUpstreamConfigs(const std::string& host, uint16_t port) {
        // Step 1: peer's local config -> upstream entry (re-keyed).
        if (!_fetch_and_insert_peer_local(host, port)) {
            return;
        }

        // Steps 2-4: pull the peer's tracker categories. Each category has
        // its own (host, port) identity baked into the hash, so entries are
        // inserted as-is and validated against the advertised hash.
        _pull_categorized_from_peer(host, port, "/config_tracker_upstream_hashes",
                                    "/config_tracker_upstream_configs", _upstream_configs,
                                    _upstream_config_hashes, "upstream", _config_present_metric());
        _pull_categorized_from_peer(
            host, port, "/config_tracker_fpga_config_hashes", "/config_tracker_fpga_configs",
            _fpga_configs, _fpga_config_hashes, "fpga config", _fpga_config_present_metric());
        _pull_categorized_from_peer(
            host, port, "/config_tracker_fpga_timing_hashes", "/config_tracker_fpga_timings",
            _fpga_timings, _fpga_timing_hashes, "fpga timing", _fpga_timing_present_metric());

        std::lock_guard<std::mutex> lock(_lock);
        _check_consistent_locked(_upstream_configs, _upstream_config_hashes, "upstream");
        _check_consistent_locked(_fpga_configs, _fpga_config_hashes, "fpga config");
        _check_consistent_locked(_fpga_timings, _fpga_timing_hashes, "fpga timing");
    }

    /**
     * @brief Get a vector of strings of all config json (local + upstream + FPGA).
     */
    std::vector<std::string> getAllJSONConfigs() const {
        std::vector<std::string> configs;
        std::lock_guard<std::mutex> lock(_lock);
        if (_local_config.has_value()) {
            configs.push_back(_local_config->to_json().dump(
                4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        for (const auto& [_, info] : _upstream_configs) {
            configs.push_back(
                info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        for (const auto& [_, info] : _fpga_configs) {
            configs.push_back(
                info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        for (const auto& [_, info] : _fpga_timings) {
            configs.push_back(
                info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        return configs;
    }

    /**
     * @brief Write all configuration data (local + upstream + FPGA) to disk.
     *
     * Filenames:
     *   local                 -> "local.json"
     *   upstream peer         -> "<host>_<port>.json"
     *   fpga config (per ctrl) -> "<host>_<port>_fpga_config.json"
     *   fpga timing (per ctrl) -> "<host>_<port>_fpga_timing.json"
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
            _write_one_atomic(directory + "/local.json", *_local_config);
            ++written;
        }

        for (const auto& [host_port, ci] : _upstream_configs) {
            _write_one_atomic(_disk_filename(directory, host_port, ""), ci);
            ++written;
        }
        for (const auto& [host_port, ci] : _fpga_configs) {
            _write_one_atomic(_disk_filename(directory, host_port, "_fpga_config"), ci);
            ++written;
        }
        for (const auto& [host_port, ci] : _fpga_timings) {
            _write_one_atomic(_disk_filename(directory, host_port, "_fpga_timing"), ci);
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
        std::vector<std::tuple<std::string, uint16_t, std::string>> cleared_fpga_config;
        std::vector<std::tuple<std::string, uint16_t, std::string>> cleared_fpga_timing;
        std::string cleared_local_hash;
        {
            std::lock_guard<std::mutex> lock(_lock);
            for (const auto& [hp, info] : _upstream_configs)
                cleared_upstream.emplace_back(hp.host, hp.port, info.json_hash);
            for (const auto& [hp, info] : _fpga_configs)
                cleared_fpga_config.emplace_back(hp.host, hp.port, info.json_hash);
            for (const auto& [hp, info] : _fpga_timings)
                cleared_fpga_timing.emplace_back(hp.host, hp.port, info.json_hash);
            if (_local_config.has_value())
                cleared_local_hash = _local_config->json_hash;

            _upstream_configs.clear();
            _upstream_config_hashes.clear();
            _fpga_configs.clear();
            _fpga_config_hashes.clear();
            _fpga_timings.clear();
            _fpga_timing_hashes.clear();
            _local_config.reset();
            _tracker_hash.clear();
            _configs_total_metric().set(0.0);
        }

        for (const auto& [host, port, hash] : cleared_upstream)
            _config_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
        for (const auto& [host, port, hash] : cleared_fpga_config)
            _fpga_config_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
        for (const auto& [host, port, hash] : cleared_fpga_timing)
            _fpga_timing_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
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
    std::map<std::string, HostPort> _upstream_config_hashes;

    /// FPGA controller startup config, keyed by the controller's REST
    /// (host, port). Hash includes (host, port).
    std::map<HostPort, ConfigInfo> _fpga_configs;
    std::map<std::string, HostPort> _fpga_config_hashes;

    /// FPGA controller startup timing snapshot, keyed by the controller's
    /// REST (host, port). Hash includes (host, port).
    std::map<HostPort, ConfigInfo> _fpga_timings;
    std::map<std::string, HostPort> _fpga_timing_hashes;

    /// Combined hash of all configurations (local first if present, then
    /// upstream + fpga_config + fpga_timing hashes in sorted order).
    std::string _tracker_hash;

    mutable std::mutex _lock;

    static constexpr const char* _metrics_stage_name = "config_tracker";

    /// MD5 of a JSON config alone (no host:port). Used for the local config.
    std::string _jsonHashLocal(const nlohmann::json& filtered_json) const {
        if (_has_kotekan_update_endpoint(filtered_json)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: _jsonHashLocal called with kotekan_update_endpoint present.");
        }
        return _md5_hex(
            filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict));
    }

    /**
     * @brief MD5 of (host, port, JSON content) so two distinct origin
     * endpoints with identical content hash to distinct values. Used for
     * upstream-peer configs and FPGA config/timing entries. Caller must have
     * stripped any kotekan_update_endpoint blocks.
     */
    std::string _jsonHashWithEndpoint(const nlohmann::json& filtered_json, const std::string& host,
                                      uint16_t port) const {
        if (_has_kotekan_update_endpoint(filtered_json)) {
            FATAL_ERROR_NON_OO("ConfigTracker: _jsonHashWithEndpoint called with "
                               "kotekan_update_endpoint present.");
        }
        std::stringstream ss;
        ss << host << ":" << port << "|"
           << filtered_json.dump(-1, '\0', false, nlohmann::json::error_handler_t::strict);
        return _md5_hex(ss.str());
    }

    /**
     * @brief Build a ConfigInfo for an FPGA entry (config or timing).
     *
     * Strips kotekan_update_endpoint blocks (defensive — FPGA payloads
     * shouldn't contain these), computes the hash, and leaves the kotekan_*
     * version metadata empty.
     */
    ConfigInfo _build_fpga_info(const std::string& host, uint16_t port,
                                const nlohmann::json& json) const {
        nlohmann::json filtered = _strip_update_endpoints(json);
        std::string json_hash = _jsonHashWithEndpoint(filtered, host, port);
        return ConfigInfo(filtered, json_hash, "", "", "", "");
    }

    /**
     * @brief Insert a ConfigInfo into a category's (HostPort -> info) +
     * (hash -> HostPort) pair of maps.
     *
     * - Validates host (IPv4 or "localhost") and port (non-zero).
     * - Idempotent on identical content; FATAL on conflict.
     * - Updates the per-category presence metric and the total-configs gauge,
     *   and refreshes the combined tracker hash on actual insertion.
     */
    void _insertCategorized(std::map<HostPort, ConfigInfo>& target,
                            std::map<std::string, HostPort>& reverse, std::string host,
                            uint16_t port, ConfigInfo info, const char* category_name,
                            prometheus::MetricFamily<prometheus::Gauge>& presence_metric) {
        if (_has_kotekan_update_endpoint(info.config)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: {} insert called with kotekan_update_endpoint present.",
                category_name);
        }

        if (host == "localhost")
            host = "127.0.0.1";

        struct sockaddr_in sa4;
        if (inet_pton(AF_INET, host.c_str(), &(sa4.sin_addr)) != 1) {
            FATAL_ERROR_NON_OO("ConfigTracker: {} insert called with invalid IPv4 address: {}",
                               category_name, host);
        }
        if (port == 0) {
            FATAL_ERROR_NON_OO("ConfigTracker: {} insert called with invalid port: {}",
                               category_name, port);
        }

        HostPort host_port{host, port};
        bool inserted = false;
        bool hash_changed = false;
        {
            std::lock_guard<std::mutex> lock(_lock);
            auto it = target.find(host_port);
            if (it != target.end()) {
                if (!_config_info_matches(it->second, info)) {
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: conflicting {} content present for host: {}, port: {}",
                        category_name, host, port);
                }
                return; // identical content; no-op
            }
            target.emplace(host_port, info);
            reverse.emplace(info.json_hash, host_port);
            presence_metric.labels({host_port.host, std::to_string(host_port.port), info.json_hash})
                .set(1.0);
            _refresh_count_metric_locked();
            // Recompute the combined tracker hash in the same critical section
            // as the map update so readers never observe new entries with a
            // stale tracker hash.
            hash_changed = _recomputeTrackerHashLocked();
            inserted = true;
        }

        if (inserted) {
            if (hash_changed) {
                _hash_changes_total().inc();
                _last_change_timestamp().set(_now_seconds());
            }
            DEBUG_NON_OO("ConfigTracker: inserted {} for {}:{}, hash: {}", category_name, host,
                         port, info.json_hash);
        }
    }

    /**
     * @brief Step 1 of getUpstreamConfigs: pull /config_tracker_local from
     * (host, port) and re-key it under that (host, port) as an upstream entry.
     *
     * Returns true on success (or "already had it"), false if the network
     * call failed or the response was malformed.
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
        peer_local.json_hash = _jsonHashWithEndpoint(filtered, host, port);

        _insertCategorized(_upstream_configs, _upstream_config_hashes, host, port, peer_local,
                           "upstream", _config_present_metric());
        _record_upstream_fetch(host, port, true);
        return true;
    }

    /**
     * @brief Pull a categorized tracker section from a peer.
     *
     * GETs the peer's hash-index endpoint, then for each missing hash GETs
     * the matching entry from configs_endpoint and inserts it under the
     * (host, port) the peer reported. Used by getUpstreamConfigs for
     * upstream-peer entries and both FPGA categories.
     *
     * Conflicts (same hash, different (host, port)) are fatal. Network and
     * parse errors on individual entries are logged and counted but not
     * fatal.
     */
    void _pull_categorized_from_peer(const std::string& peer_host, uint16_t peer_port,
                                     const std::string& hashes_endpoint,
                                     const std::string& configs_endpoint,
                                     std::map<HostPort, ConfigInfo>& target,
                                     std::map<std::string, HostPort>& reverse,
                                     const char* category_name,
                                     prometheus::MetricFamily<prometheus::Gauge>& presence_metric) {
        restClient::restReply reply = restClient::instance().make_request_blocking(
            hashes_endpoint, {}, peer_host, peer_port, 1, -1);
        if (!reply.first) {
            ERROR_NON_OO("ConfigTracker: failed to GET {} from {}:{}", hashes_endpoint, peer_host,
                         peer_port);
            _record_upstream_fetch(peer_host, peer_port, false);
            return;
        }

        nlohmann::json hashes_json;
        try {
            hashes_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            ERROR_NON_OO("ConfigTracker: failed to parse {} response from {}:{}: {}",
                         hashes_endpoint, peer_host, peer_port, e.what());
            DEBUG2_NON_OO("Response was: {}", reply.second);
            _record_upstream_fetch(peer_host, peer_port, false);
            return;
        }
        _record_upstream_fetch(peer_host, peer_port, true);

        for (const auto& item : hashes_json.items()) {
            const std::string& hash = item.key();
            const nlohmann::json& host_port_json = item.value();
            std::string entry_host = host_port_json.value("host", peer_host);
            uint16_t entry_port = host_port_json.value("port", peer_port);

            {
                std::lock_guard<std::mutex> lock(_lock);
                auto it = reverse.find(hash);
                if (it != reverse.end()) {
                    if (it->second.host == entry_host && it->second.port == entry_port)
                        continue;
                    FATAL_ERROR_NON_OO("ConfigTracker: {} hash {} already known for {}:{}, but "
                                       "peer {}:{} reports it as {}:{}",
                                       category_name, hash, it->second.host, it->second.port,
                                       peer_host, peer_port, entry_host, entry_port);
                }
            }

            // Fetch the entry by hash from the peer.
            const std::string path = configs_endpoint + "?hash=" + hash;
            restClient::restReply entry_reply =
                restClient::instance().make_request_blocking(path, {}, peer_host, peer_port, 1, -1);
            if (!entry_reply.first) {
                ERROR_NON_OO("ConfigTracker: failed to GET {} for {} hash {} from peer {}:{}",
                             configs_endpoint, category_name, hash, peer_host, peer_port);
                _record_upstream_fetch(entry_host, entry_port, false);
                continue;
            }

            nlohmann::json response;
            try {
                response = nlohmann::json::parse(entry_reply.second);
            } catch (const nlohmann::json::parse_error& e) {
                ERROR_NON_OO("ConfigTracker: failed to parse {} response for {} hash {}: {}",
                             configs_endpoint, category_name, hash, e.what());
                _record_upstream_fetch(entry_host, entry_port, false);
                continue;
            }

            const std::string host_port_str = entry_host + ":" + std::to_string(entry_port);
            if (!response.contains(host_port_str)) {
                ERROR_NON_OO("ConfigTracker: peer {}:{} did not return {} for {} (hash {})",
                             peer_host, peer_port, category_name, host_port_str, hash);
                _record_upstream_fetch(entry_host, entry_port, false);
                continue;
            }

            ConfigInfo info;
            try {
                info = ConfigInfo(response[host_port_str]);
            } catch (const std::exception& e) {
                ERROR_NON_OO("ConfigTracker: malformed {} payload for {} (hash {}): {}",
                             category_name, host_port_str, hash, e.what());
                _record_upstream_fetch(entry_host, entry_port, false);
                continue;
            }

            if (_jsonHashWithEndpoint(info.config, entry_host, entry_port) != hash
                || info.json_hash != hash) {
                ERROR_NON_OO("ConfigTracker: {} hash mismatch for {} (expected {}); dropping",
                             category_name, host_port_str, hash);
                _record_upstream_fetch(entry_host, entry_port, false);
                continue;
            }

            _insertCategorized(target, reverse, entry_host, entry_port, info, category_name,
                               presence_metric);
            _record_upstream_fetch(entry_host, entry_port, true);
        }
    }

    /**
     * @brief REST helper: serve a category's entries (optionally filtered by ?hash=).
     */
    void _serve_categorized_entries(connectionInstance& conn,
                                    const std::map<HostPort, ConfigInfo>& entries,
                                    const std::map<std::string, HostPort>& reverse,
                                    const char* category_name) {
        std::lock_guard<std::mutex> lock(_lock);
        _check_consistent_locked(entries, reverse, category_name);

        nlohmann::json return_json = nlohmann::json::object();

        auto query_args = conn.get_query();
        std::string hash;
        if (query_args.find("hash") != query_args.end())
            hash = query_args["hash"];

        for (const auto& [host_port, info] : entries) {
            if (!hash.empty() && info.json_hash != hash)
                continue;
            std::string host_port_str = host_port.host + ":" + std::to_string(host_port.port);
            return_json[host_port_str] = info.to_json();
        }
        conn.send_json_reply(return_json);
    }

    /**
     * @brief REST helper: serve a category's hash -> {host, port} index.
     */
    void _serve_categorized_hashes(connectionInstance& conn,
                                   const std::map<HostPort, ConfigInfo>& entries,
                                   const std::map<std::string, HostPort>& reverse,
                                   const char* category_name) {
        std::lock_guard<std::mutex> lock(_lock);
        _check_consistent_locked(entries, reverse, category_name);

        nlohmann::json return_json = nlohmann::json::object();
        for (const auto& [hash, host_port] : reverse) {
            return_json[hash] = {{"host", host_port.host}, {"port", host_port.port}};
        }
        conn.send_json_reply(return_json);
    }

    /**
     * @brief Recompute and store the combined tracker hash from current map
     * state. Caller must hold @c _lock.
     *
     * The hash composes: local hash (if present), then sorted upstream + FPGA
     * config + FPGA timing hashes. The order is fixed so the combined hash is
     * deterministic regardless of insertion order. Computing inside the
     * existing critical section guarantees that any reader observing a new
     * entry also observes the matching tracker hash.
     *
     * @return true if @c _tracker_hash actually changed value.
     */
    bool _recomputeTrackerHashLocked() {
        std::stringstream ss;
        if (_local_config.has_value())
            ss << _local_config->json_hash;
        for (const auto& [hash, _] : _upstream_config_hashes)
            ss << hash;
        for (const auto& [hash, _] : _fpga_config_hashes)
            ss << hash;
        for (const auto& [hash, _] : _fpga_timing_hashes)
            ss << hash;

        const std::string new_hash = _md5_hex(ss.str());
        bool changed = (_tracker_hash != new_hash);
        _tracker_hash = new_hash;
        DEBUG_NON_OO("ConfigTracker: combined tracker hash is now {}", _tracker_hash);
        return changed;
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

    /// Build "<directory>/<host>_<port><suffix>.json".
    static std::string _disk_filename(const std::string& directory, const HostPort& hp,
                                      const std::string& suffix) {
        std::ostringstream ss;
        ss << directory << "/" << hp.host << "_" << hp.port << suffix << ".json";
        return ss.str();
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
                file << info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict);
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

    /// Sanity check on a category's two maps. Caller must hold _lock.
    void _check_consistent_locked(const std::map<HostPort, ConfigInfo>& entries,
                                  const std::map<std::string, HostPort>& reverse,
                                  const char* category_name) const {
        if (entries.size() != reverse.size()) {
            FATAL_ERROR_NON_OO("ConfigTracker: {} entries ({}) and hashes ({}) sizes differ",
                               category_name, entries.size(), reverse.size());
        }
    }

    /// Refresh the total-configs gauge. Caller must hold _lock.
    void _refresh_count_metric_locked() {
        _configs_total_metric().set(
            static_cast<double>((_local_config.has_value() ? 1u : 0u) + _upstream_configs.size()
                                + _fpga_configs.size() + _fpga_timings.size()));
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
            prometheus::Metrics::instance().add_gauge("kotekan_config_tracker_local_config_present",
                                                      _metrics_stage_name, {"hash"});
        return metric;
    }

    static prometheus::MetricFamily<prometheus::Gauge>& _fpga_config_present_metric() {
        static prometheus::MetricFamily<prometheus::Gauge>& metric =
            prometheus::Metrics::instance().add_gauge("kotekan_config_tracker_fpga_config_present",
                                                      _metrics_stage_name,
                                                      {"host", "port", "hash"});
        return metric;
    }

    static prometheus::MetricFamily<prometheus::Gauge>& _fpga_timing_present_metric() {
        static prometheus::MetricFamily<prometheus::Gauge>& metric =
            prometheus::Metrics::instance().add_gauge("kotekan_config_tracker_fpga_timing_present",
                                                      _metrics_stage_name,
                                                      {"host", "port", "hash"});
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
