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
 * An upstream FPGA controller, if present, is registered as a regular
 * upstream entry: a single combined ``{"config": ..., "timing": ...}`` JSON,
 * keyed by the controller's REST (host, port). Any change to either part
 * after the initial fetch indicates a controller reset and is fatal.
 *
 * - kotekan::ConfigTracker
 *   -- instance
 *   -- n_configs
 *   -- getTrackerHash
 *   -- setLocalConfig
 *   -- insertUpstreamConfig
 *   -- fetchAndRegisterFpgaTracking
 *   -- hasLocalConfig / hasUpstreamConfig
 *   -- getLocalConfigInfo / getLocalConfigHash
 *   -- trackers_local_callback
 *   -- trackers_local_hash_callback
 *   -- trackers_upstream_configs_callback / trackers_upstream_hashes_callback
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

#include <arpa/inet.h>   // for inet_pton, inet_ntop
#include <chrono>        // for duration, duration_cast, system_clock
#include <cstdio>        // for remove, rename, size_t
#include <errno.h>       // for errno
#include <exception>     // for exception
#include <fstream>       // for basic_ofstream
#include <functional>    // for bind, _1, function
#include <iomanip>       // for operator<<, setfill, setw
#include <map>           // for map, operator!=, _Rb_tree_iterator, _Rb_tree_const_iterator
#include <mutex>         // for mutex, lock_guard
#include <netdb.h>       // for getaddrinfo, freeaddrinfo, gai_strerror
#include <netinet/in.h>  // for sockaddr_in
#include <openssl/md5.h> // for MD5, MD5_DIGEST_LENGTH
#include <optional>      // for optional
#include <sstream>       // for basic_ostream, basic_stringstream, operator<<, basic_ostre...
#include <stdint.h>      // for uint16_t
#include <string.h>      // for strerror
#include <string>        // for basic_string, allocator, char_traits, operator+, string
#include <sys/socket.h>  // for AF_INET, SOCK_STREAM
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
     * For non-kotekan upstream entries (e.g., an FPGA controller) the
     * ``kotekan_*`` version fields are unused and stored as empty strings.
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
     * @brief Insert a config received from an upstream peer (or from any
     * upstream JSON-emitting source like an FPGA controller) into the tracker.
     *
     * The upstream entry is keyed by (host, port). The hash bakes in the
     * (host, port) so that two distinct origins with identical content still
     * produce distinct hashes (preserving the 1:1 invariant with
     * _upstream_config_hashes).
     *
     * @param host         The (ipv4) host of the upstream, as observed by
     *                     this node (i.e., the address this node dialed).
     * @param port         The REST port of the upstream.
     * @param config_json  The JSON content to insert.
     * @param kotekan_version       Kotekan version (empty for non-kotekan upstreams).
     * @param kotekan_build_branch  Kotekan build branch (empty for non-kotekan upstreams).
     * @param kotekan_git_commit_hash    Kotekan git commit hash (empty for non-kotekan upstreams).
     * @param kotekan_cmake_options Kotekan CMake options (empty for non-kotekan upstreams).
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

        _insertUpstream(host, port, info);
    }

    /**
     * @brief Set the retry/timeout policy used by both
     * @c fetchAndRegisterFpgaTracking (the one-shot FPGA fetch at startup)
     * and @c getUpstreamConfigs (peer fetches on every wire-update flag).
     * Intended to be called once at kotekan startup from kotekanMode after
     * reading the /config_tracker block. After retries are exhausted on any
     * fetch, the call is fatal.
     *
     * @param retries          Number of retries for each upstream
     *                         @c make_request_blocking call.
     * @param timeout_seconds  HTTP timeout in seconds (-1 = restClient default).
     */
    void setUpstreamFetchPolicy(int retries, int timeout_seconds) {
        std::lock_guard<std::mutex> lock(_lock);
        _upstream_fetch_retries = retries;
        _upstream_fetch_timeout_seconds = timeout_seconds;
    }

    /**
     * @brief Fetch the upstream FPGA controller's config and timing JSON via
     * REST and register them as a single upstream entry. Intended to be
     * called once at kotekan startup (see kotekanMode).
     *
     * The hostname is resolved to a canonical IPv4 string up front; that
     * IPv4 is used both as the @c host argument to @c restClient and as the
     * storage key, so downstream peers fetching the entry transitively land
     * on the same key. The returned config and timing payloads are wrapped
     * into a single ``{"config": ..., "timing": ...}`` JSON and registered
     * via @c insertUpstreamConfig with empty kotekan_* metadata, so the FPGA
     * snapshot rides along with peer configs through the normal upstream
     * propagation path.
     *
     * Failure to resolve the hostname, to issue either GET, or to parse
     * either response is fatal: kotekan will not start without a confirmed
     * FPGA controller snapshot when this is called.
     *
     * Uses the retry/timeout policy set via @c setUpstreamFetchPolicy.
     *
     * @param host             Hostname or IPv4 address of the FPGA
     *                         controller's REST server.
     * @param port             REST port.
     * @param config_endpoint  Path to the controller's config JSON.
     * @param timing_endpoint  Path to the controller's timing JSON.
     */
    void fetchAndRegisterFpgaTracking(const std::string& host, uint16_t port,
                                      const std::string& config_endpoint,
                                      const std::string& timing_endpoint) {
        const std::string ip = _resolveToIPv4(host);
        if (host != ip) {
            INFO_NON_OO("ConfigTracker: resolved FPGA controller {}:{} to {}:{}", host, port, ip,
                        port);
        }

        int retries;
        int timeout_seconds;
        {
            std::lock_guard<std::mutex> lock(_lock);
            retries = _upstream_fetch_retries;
            timeout_seconds = _upstream_fetch_timeout_seconds;
        }

        nlohmann::json cfg_json =
            _fpga_get_or_fatal(ip, port, config_endpoint, "config", timeout_seconds, retries);
        nlohmann::json tim_json =
            _fpga_get_or_fatal(ip, port, timing_endpoint, "timing", timeout_seconds, retries);

        // Combine config + timing into a single upstream entry. Storage and
        // propagation are uniform with peer kotekan configs from here on.
        nlohmann::json combined = {{"config", cfg_json}, {"timing", tim_json}};
        insertUpstreamConfig(ip, port, combined, "", "", "", "");

        INFO_NON_OO(
            "ConfigTracker: registered FPGA controller config and timing from {}:{} (config {}, "
            "timing {})",
            ip, port, config_endpoint, timing_endpoint);
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

    /// REST callback returning upstream configs (optionally filtered by ?hash=).
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

    /// REST callback returning upstream-config hash -> {host, port}.
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
     *   GET /config_tracker_upstream_configs -> upstream configs (or ?hash=)
     *   GET /config_tracker_upstream_hashes  -> {hash: {host, port}} for upstream
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
    }

    /**
     * @brief Fetch and insert an upstream peer's tracker state.
     *
     * Two-step protocol against the peer at (host, port):
     *   1. GET /config_tracker_local. The returned ConfigInfo is the peer's
     *      own local config; this node re-keys it under (host, port) (the
     *      address it actually dialed) and stores it as an upstream entry.
     *   2. GET /config_tracker_upstream_hashes. For each hash not already
     *      present locally, GET /config_tracker_upstream_configs?hash=<...>
     *      and insert it (its host:port reflects what the peer observed,
     *      which we trust transitively). FPGA snapshots are ordinary
     *      upstream entries and ride along on this same path.
     *
     * Any network/parse failure (after the configured HTTP retries) or
     * any content-validation failure is fatal. The wire flag is one-shot
     * per sender-side hash change, so a silently-skipped fetch would leave
     * downstream state permanently stale.
     */
    void getUpstreamConfigs(const std::string& host, uint16_t port) {
        // Snapshot the fetch policy under the lock; we'll use it for all
        // three peer fetches in this invocation.
        int retries;
        int timeout_seconds;
        {
            std::lock_guard<std::mutex> lock(_lock);
            retries = _upstream_fetch_retries;
            timeout_seconds = _upstream_fetch_timeout_seconds;
        }

        // Step 1: GET /config_tracker_local from the peer; re-key under the
        // dialed (host, port) and store as an upstream entry.
        {
            restClient::restReply reply = restClient::instance().make_request_blocking(
                "/config_tracker_local", {}, host, port, retries, timeout_seconds);
            if (!reply.first) {
                _record_upstream_fetch(host, port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: failed to GET /config_tracker_local from {}:{} after {} "
                    "retries",
                    host, port, retries);
            }

            nlohmann::json local_json;
            try {
                local_json = nlohmann::json::parse(reply.second);
            } catch (const nlohmann::json::parse_error& e) {
                _record_upstream_fetch(host, port, false);
                DEBUG2_NON_OO("Response was: {}", reply.second);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: failed to parse /config_tracker_local response from {}:{}: {}",
                    host, port, e.what());
            }

            ConfigInfo peer_local;
            try {
                peer_local = ConfigInfo(local_json);
            } catch (const std::exception& e) {
                _record_upstream_fetch(host, port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: malformed /config_tracker_local payload from {}:{}: {}", host,
                    port, e.what());
            }

            // Re-key under the dialed (host, port) and re-hash accordingly.
            nlohmann::json filtered = _strip_update_endpoints(peer_local.config);
            peer_local.config = filtered;
            peer_local.json_hash = _jsonHashWithEndpoint(filtered, host, port);

            _insertUpstream(host, port, peer_local);
            _record_upstream_fetch(host, port, true);
        }

        // Step 2: GET /config_tracker_upstream_hashes; for each missing hash
        // pull /config_tracker_upstream_configs?hash=<...> and insert.
        // Entries (including FPGA snapshots) carry their own (host, port)
        // identity from the peer; we trust that transitively.
        restClient::restReply reply = restClient::instance().make_request_blocking(
            "/config_tracker_upstream_hashes", {}, host, port, retries, timeout_seconds);
        if (!reply.first) {
            _record_upstream_fetch(host, port, false);
            FATAL_ERROR_NON_OO(
                "ConfigTracker: failed to GET /config_tracker_upstream_hashes from {}:{} after "
                "{} retries",
                host, port, retries);
        }

        nlohmann::json hashes_json;
        try {
            hashes_json = nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            _record_upstream_fetch(host, port, false);
            DEBUG2_NON_OO("Response was: {}", reply.second);
            FATAL_ERROR_NON_OO("ConfigTracker: failed to parse /config_tracker_upstream_hashes "
                               "response from {}:{}: {}",
                               host, port, e.what());
        }
        _record_upstream_fetch(host, port, true);

        for (const auto& item : hashes_json.items()) {
            const std::string& hash = item.key();
            const nlohmann::json& host_port_json = item.value();
            std::string entry_host = host_port_json.value("host", host);
            uint16_t entry_port = host_port_json.value("port", port);

            {
                std::lock_guard<std::mutex> lock(_lock);
                auto it = _upstream_config_hashes.find(hash);
                if (it != _upstream_config_hashes.end()) {
                    if (it->second.host == entry_host && it->second.port == entry_port)
                        continue;
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: upstream hash {} already known for {}:{}, but peer {}:{} "
                        "reports it as {}:{}",
                        hash, it->second.host, it->second.port, host, port, entry_host, entry_port);
                }
            }

            const std::string path = std::string("/config_tracker_upstream_configs?hash=") + hash;
            restClient::restReply entry_reply = restClient::instance().make_request_blocking(
                path, {}, host, port, retries, timeout_seconds);
            if (!entry_reply.first) {
                _record_upstream_fetch(entry_host, entry_port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: failed to GET /config_tracker_upstream_configs for hash {} "
                    "from peer {}:{} after {} retries",
                    hash, host, port, retries);
            }

            nlohmann::json response;
            try {
                response = nlohmann::json::parse(entry_reply.second);
            } catch (const nlohmann::json::parse_error& e) {
                _record_upstream_fetch(entry_host, entry_port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: failed to parse upstream-config response for hash {}: {}", hash,
                    e.what());
            }

            const std::string host_port_str = entry_host + ":" + std::to_string(entry_port);
            if (!response.contains(host_port_str)) {
                _record_upstream_fetch(entry_host, entry_port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: peer {}:{} did not return upstream entry for {} (hash {})",
                    host, port, host_port_str, hash);
            }

            ConfigInfo info;
            try {
                info = ConfigInfo(response[host_port_str]);
            } catch (const std::exception& e) {
                _record_upstream_fetch(entry_host, entry_port, false);
                FATAL_ERROR_NON_OO(
                    "ConfigTracker: malformed upstream-config payload for {} (hash {}): {}",
                    host_port_str, hash, e.what());
            }

            if (_jsonHashWithEndpoint(info.config, entry_host, entry_port) != hash
                || info.json_hash != hash) {
                _record_upstream_fetch(entry_host, entry_port, false);
                FATAL_ERROR_NON_OO("ConfigTracker: upstream hash mismatch for {} (expected {})",
                                   host_port_str, hash);
            }

            _insertUpstream(entry_host, entry_port, info);
            _record_upstream_fetch(entry_host, entry_port, true);
        }

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
            configs.push_back(_local_config->to_json().dump(
                4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        for (const auto& [_, info] : _upstream_configs) {
            configs.push_back(
                info.to_json().dump(4, ' ', false, nlohmann::json::error_handler_t::strict));
        }
        return configs;
    }

    /**
     * @brief Write all configuration data (local + upstream) to disk.
     *
     * Filenames:
     *   local         -> "local.json"
     *   upstream      -> "<host>_<port>.json"
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
            std::ostringstream filename_stream;
            filename_stream << directory << "/" << host_port.host << "_" << host_port.port
                            << ".json";
            _write_one_atomic(filename_stream.str(), ci);
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
            for (const auto& [hp, info] : _upstream_configs)
                cleared_upstream.emplace_back(hp.host, hp.port, info.json_hash);
            if (_local_config.has_value())
                cleared_local_hash = _local_config->json_hash;

            _upstream_configs.clear();
            _upstream_config_hashes.clear();
            _local_config.reset();
            _tracker_hash.clear();
            _configs_total_metric().set(0.0);
        }

        for (const auto& [host, port, hash] : cleared_upstream)
            _config_present_metric().labels({host, std::to_string(port), hash}).set(0.0);
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

    /// Configs received from upstream peers (and from non-kotekan upstream
    /// snapshots like the FPGA controller), keyed by the (host, port) this
    /// node observed for the upstream. Hash includes (host, port).
    std::map<HostPort, ConfigInfo> _upstream_configs;
    std::map<std::string, HostPort> _upstream_config_hashes;

    /// Combined hash of all configurations (local first if present, then
    /// upstream hashes in sorted order).
    std::string _tracker_hash;

    /// HTTP retry/timeout policy for all upstream fetches: the one-shot FPGA
    /// controller fetch in @c fetchAndRegisterFpgaTracking and every peer
    /// fetch in @c getUpstreamConfigs. Set once at startup via
    /// @c setUpstreamFetchPolicy.
    int _upstream_fetch_retries = 2;
    int _upstream_fetch_timeout_seconds = 10;

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
     * upstream entries (peer kotekan configs and FPGA snapshots alike).
     * Caller must have stripped any kotekan_update_endpoint blocks.
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
     * @brief Insert a ConfigInfo as an upstream entry under (host, port).
     *
     * - Validates host (IPv4 or "localhost") and port (non-zero).
     * - Idempotent on identical content; FATAL on conflict.
     * - Updates the upstream-presence metric and the total-configs gauge,
     *   and refreshes the combined tracker hash on actual insertion.
     */
    void _insertUpstream(std::string host, uint16_t port, ConfigInfo info) {
        if (_has_kotekan_update_endpoint(info.config)) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: upstream insert called with kotekan_update_endpoint present.");
        }

        if (host == "localhost")
            host = "127.0.0.1";

        struct sockaddr_in sa4;
        if (inet_pton(AF_INET, host.c_str(), &(sa4.sin_addr)) != 1) {
            FATAL_ERROR_NON_OO(
                "ConfigTracker: upstream insert called with invalid IPv4 address: {}", host);
        }
        if (port == 0) {
            FATAL_ERROR_NON_OO("ConfigTracker: upstream insert called with invalid port: {}", port);
        }

        HostPort host_port{host, port};
        bool inserted = false;
        bool hash_changed = false;
        {
            std::lock_guard<std::mutex> lock(_lock);
            auto it = _upstream_configs.find(host_port);
            if (it != _upstream_configs.end()) {
                if (!_config_info_matches(it->second, info)) {
                    FATAL_ERROR_NON_OO(
                        "ConfigTracker: conflicting upstream content present for host: {}, "
                        "port: {}",
                        host, port);
                }
                return; // identical content; no-op
            }
            _upstream_configs.emplace(host_port, info);
            _upstream_config_hashes.emplace(info.json_hash, host_port);
            _config_present_metric()
                .labels({host_port.host, std::to_string(host_port.port), info.json_hash})
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
            DEBUG_NON_OO("ConfigTracker: inserted upstream for {}:{}, hash: {}", host, port,
                         info.json_hash);
        }
    }

    /**
     * @brief Recompute and store the combined tracker hash from current map
     * state. Caller must hold @c _lock.
     *
     * The hash composes the local hash (if present), then the sorted upstream
     * hashes. The order is fixed so the combined hash is deterministic
     * regardless of insertion order. Computing inside the existing critical
     * section guarantees that any reader observing a new entry also observes
     * the matching tracker hash.
     *
     * @return true if @c _tracker_hash actually changed value.
     */
    bool _recomputeTrackerHashLocked() {
        std::stringstream ss;
        if (_local_config.has_value())
            ss << _local_config->json_hash;
        for (const auto& [hash, _] : _upstream_config_hashes)
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

    /**
     * @brief Resolve a hostname-or-IPv4 string to its canonical IPv4
     * representation. Fatal on resolution failure.
     */
    static std::string _resolveToIPv4(const std::string& host_or_ip) {
        // If it already parses as IPv4, return as-is.
        struct sockaddr_in sa4;
        if (inet_pton(AF_INET, host_or_ip.c_str(), &(sa4.sin_addr)) == 1)
            return host_or_ip;

        struct addrinfo hints = {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        struct addrinfo* res = nullptr;
        const int err = getaddrinfo(host_or_ip.c_str(), nullptr, &hints, &res);
        if (err != 0 || res == nullptr) {
            FATAL_ERROR_NON_OO("ConfigTracker: failed to resolve host '{}': {}", host_or_ip,
                               gai_strerror(err));
        }

        char buf[INET_ADDRSTRLEN] = {0};
        auto* sa = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
        inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf));
        freeaddrinfo(res);
        return std::string(buf);
    }

    /**
     * @brief GET a single FPGA controller endpoint and return the parsed JSON
     * body. Fatal on HTTP failure (after retries) or parse failure.
     */
    static nlohmann::json _fpga_get_or_fatal(const std::string& ip, uint16_t port,
                                             const std::string& endpoint, const char* what,
                                             int request_timeout_seconds, int request_retries) {
        auto reply = restClient::instance().make_request_blocking(
            endpoint, {}, ip, port, request_retries, request_timeout_seconds);
        if (!reply.first) {
            FATAL_ERROR_NON_OO("ConfigTracker: failed to GET FPGA {} from {}:{}{} after {} retries "
                               "(timeout {}s per attempt)",
                               what, ip, port, endpoint, request_retries, request_timeout_seconds);
        }
        try {
            return nlohmann::json::parse(reply.second);
        } catch (const nlohmann::json::parse_error& e) {
            FATAL_ERROR_NON_OO("ConfigTracker: failed to parse FPGA {} response from {}:{}{}: {}",
                               what, ip, port, endpoint, e.what());
        }
        // Unreachable; FATAL_ERROR_NON_OO throws.
        return nlohmann::json{};
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

    /// Sanity check that _upstream_configs and _upstream_config_hashes have
    /// the same size. Caller must hold _lock.
    void _check_upstream_consistent_locked() const {
        if (_upstream_configs.size() != _upstream_config_hashes.size()) {
            FATAL_ERROR_NON_OO("ConfigTracker: upstream entries ({}) and hashes ({}) sizes differ",
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
            prometheus::Metrics::instance().add_gauge("kotekan_config_tracker_local_config_present",
                                                      _metrics_stage_name, {"hash"});
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
