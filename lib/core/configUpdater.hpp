#ifndef CONFIGUPDATER_H
#define CONFIGUPDATER_H

#include "Config.hpp"
#include "restServer.hpp"

/**
 * @brief Kotekan core component that creates endpoints defined in the config
 * that processes can subscribe to to receive updates.
 *
 * An endpoint will be created for every updatable config block defined in the
 * configuration file. updatable blocks can be anywhere in the configuration
 * tree, but may not be inside another updatable block. They need to contain a
 * key `kotekan_update_endpoint` with the value `"json"`. They also need to
 * contain initial values for all fields that subscribing processes will expect
 * on an update.
 *
 * Example:
 * ```
 * foo:
 *     bar:
 *         kotekan_update_endpoint: "json"
 *         some_value: 0
 * my_process:
 *     updatable_config: "/foo/bar"
 * ```
 *
 * Every process that subscribes to this update endpoint by calling
 * ```
 * configUpdater::instance().subscribe(config.get_string(unique_name, "updatable_config"),
 * std::bind(&my_process::my_callback, this, _1));
 * ```
 * will receive an initial update on the callback function it implements and
 * hands over to subscribe() with the initial values defined in the config file
 * (in this case {"some_value": 0}).
 *
 * The process should check the update for correctness and return `false` if
 * the update is bad.
 */
class configUpdater
{
    public:
        /**
        * @brief Get the global configUpdater.
        *
        * @returns A reference to the global configUpdater instance.
        **/
       static configUpdater& instance();

       // Remove the implicit copy/assignments to prevent copying
       configUpdater(const configUpdater&) = delete;
       void operator=(const configUpdater&) = delete;

       /**
        * @brief Set and apply the static config to configUpdater
        * @param config         The config.
        */
       void apply_config(Config& config);

       /**
        * @brief Subscribe to a dynamic attribute.
        *
        * The callback function has to return True on success and False
        * otherwise.
        *
        * @param name       Name of the dynamic attribute.
        * @param callback   Callback function for attribute updates.
        */
       void subscribe(const string& name,
                                     std::function<bool(json &)> callback);

       /// This should be called by restServer
       void rest_callback(connectionInstance &con,
                                         nlohmann::json &json);

    private:
        /// Constructor, we don't want anyone to call this
        configUpdater() : _config(nullptr) { }

        /// Creates a new endpoint with a given name
        void create_endpoint(const string& name);

        /// Parses the config tree and calls create_endpoint when it encounters
        /// kotekan_update_endpoint in a block
        void parse_tree(json& config_tree, const string& path);

        /// unique names of endpoints that the configUpdater controlls
        vector<string> _endpoints;

        /// mmap of all subscriber callback functions for the registered dynamic
        /// attributes
        std::multimap<std::string, std::function<bool(json &)>> _callbacks;

        /// Initial values found in config yaml file
        std::map<std::string, nlohmann::json> _init_values;

        /// Reference to the Config instance in order to pass updates to it
        Config *_config;
};

#endif // CONFIGUPDATER_H
