#ifndef CONFIGMTRACKER_H
#define CONFIGMTRACKER_H

#include "Config.hpp"     // for Config
#include "Stage.hpp"      // for Stage
#include "restServer.hpp" // for connectionInstance

#include "json.hpp" // for json

#include <string>     // for string
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
    static ConfigTracker& instance();

    // Remove the implicit copy/assignments to prevent copying
    ConfigTracker(const ConfigTracker&) = delete;
    void operator=(const ConfigTracker&) = delete;

    /**
     * @brief Reset the ConfigTracker (clear the _configs)
     */
    void reset();

    /**
     * @brief Insert raw JSON + hash
     */
    void insertConfig(const nlohmann::json& config);

    /**
     * @brief Determine if a config exists in the _configs map.
     *
     * @param hash The hash of the config.
     * @returns True if the config exists, false otherwise.
     */
    bool hasConfig(const std::string& hash) const;

    /**
     * @brief Get a config from the _configs map.
     *
     * @param hash The hash of the config to retrieve.
     * @returns The config as a nlohmann::json object.
     */
    const nlohmann::json& getConfig(const std::string& hash) const;

    /**
     * @brief Attempt to fetch a config from a peer REST server and cache it.
     *
     * The default REST port is 12048 (same as the kotekan restServer).
     */
    void requestUpstreamConfig(const std::string& ip,
                               const std::string& hash,
                               uint16_t port = 12048);
    /**
     * @brief Get the canonical hash of a config.
     *
     * This function generates a consistent hash for a given configuration JSON object.
     *
     * @param config The configuration JSON object to hash.
     * @returns The canonical hash as a string.
     */
    std::string canonicalHash(const nlohmann::json& config) const;

    /**
     * @brief Get a hash of the hash of all configurations stored in the tracker.
     *
     * @returns A string hash of the combined hash of all configurations.
     */
    std::string getCombinedHash() const;

    /**
     * @brief Set a hash of the hash of all configurations stored in the tracker.
     *
     * @returns A string hash of the combined hash of all configurations.
     */
    void setCombinedHash();

    /**
     * @brief Get the number of configurations stored in the tracker.
     * @returns The number of configurations.
     */
    std::size_t n_configs() const {
        std::lock_guard<std::mutex> lock(_lock);
        return _configs.size();
    }

    /**
     * @brief Registers this class with the REST server, creating the
     *        /trackers end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server) {
        using namespace std::placeholders;
        rest_server->register_get_callback("/config_tracker",
                                           std::bind(&ConfigTracker::trackers_callback, this, _1));
    }

    /**
     * @brief The call back function for the REST server to use.
     * This returns all contents from _configs as its own JSON object.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_callback(connectionInstance& conn) {
        nlohmann::json return_json = {};

        std::lock_guard<std::mutex> lock(_lock);
        for (const auto& config : _configs) {
            return_json[config.first] = config.second;
        }

        conn.send_json_reply(return_json);
    }

private:
    /// Constructor, we don't want anyone to call this
    ConfigTracker() = default;

    /// List of hash/config pairs
    std::map<std::string, nlohmann::json> _configs;

    /// Combined hash of all configurations
    std::string _combined_hash;

    mutable std::mutex _lock;
};

} // namespace kotekan

#endif // CONFIGTRACKER_H
