#ifndef CONFIGUPDATER_H
#define CONFIGUPDATER_H

#include "restServer.hpp"
#include "KotekanProcess.hpp"
#include "Config.hpp"

#include "json.hpp"

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
 *         some_other_value: 1
 * ```
 *
 * In the config block of a process that wants to get updates of a specific
 * updatable block, either a key "updatable_config" with the full path to the updatable
 * block as a value has to exist, or an object called "updatable_config" with
 * a list of all updatable config blocks.
 *
 * Example:
 * ```
 * my_process:
 *     updatable_config: "/foo/bar"
 * ```
 * or
 * ```
 * my_process:
 *     updatable_config:
 *         bar: "/foo/bar"
 *         fu: "/foo/fu"
 * ```
 *
 * Every process that subscribes to this update endpoint by calling
 * ```
 * configUpdater config_updater = configUpdater::instance();
 * config_updater.subscribe(*this, std::bind(&my_process::my_callback, this, _1));
 * ```
 * or
 * ```
 * std::map<std::string, std::function<bool(json &)> callback_map(2);
 *
 * callbacks.insert ("bar", std::bind(&my_process::my_bar_callback, this, _1));
 *
 * callbacks.insert ("fu", std::bind(&my_process::my_fu_callback, this, _1));
 *
 * configUpdater::instance().subscribe(*this, callback_map);
 * ```
 * will receive an initial update on each callback function with the initial
 * values defined in the config file (in the first example
 * `{"some_value": 0, some_other_value: 1}`).
 * That's why the process must be ready to receive updates **before** it
 * subscribes.
 *
 * All and only the variables defined in the updatable config block in the
 * config file are guaranteed to be in the json block passed to the processes
 * callback function.
 * It is up to the process, though, to check the data types, sizes and
 * the actual values in the callback function and return `false` if anything
 * is wrong.
 *
 * The process must be ready to receive updates **before** it subscribes and it
 * has to apply save threading principles.
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
        * @brief Reset the configUpdater
        *
        * Removes all REST endpoints and clears all memory of subscribers and
        * endpoints. This should be called **before destruction of the
        * subscribers**, to prevent the callbacks being called afterwards.
        */
       void reset();

       /**
        * @brief Subscribe to the updatable blocks of a KotekanProcess.
        *
        * The callback function has to return True on success and False
        * otherwise.
        * The block of the calling process in the configuration file should have
        * a key named "updatable_block" that defines the full path to the
        * updatable block.
        *
        * @param subscriber Reference to the subscribing process.
        * @param callback   Callback function for attribute updates.
        */
       void subscribe(const KotekanProcess* subscriber,
                      std::function<bool(json &)> callback);

       /**
        * @brief Subscribe to all updatable blocks of a KotekanProcess.
        *
        * The callback functions have to return True on success and False
        * otherwise.
        * The block of the calling process in the configuration file should have
        * an object named "updatable_block" with values that define the full
        * path to an updatable block, each. The names in the callbacks map refer
        * to the names of these values.
        *
        * @param subscriber Reference to the subscribing process.
        * @param callbacks  Map of value names and callback functions.
        */
       void subscribe(const KotekanProcess* subscriber,
                   std::map<std::string,std::function<bool(json &)>> callbacks);

       /**
        * @brief Subscribe to an updatable block.
        *
        * This function does not enforce the config structure and should
        * only be used in special cases (Like when called from somewhere else
        * than a KotekanProcess).
        * The callback function has to return True on success and False
        * otherwise.
        *
        * @param name       Name of the dynamic attribute.
        * @param callback   Callback function for attribute updates.
        */
       void subscribe(const string& name, std::function<bool(json &)> callback);

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

        /// Names of the variables found in each updatable config block
        std::map<std::string, std::vector<std::string>> _keys;

        /// Reference to the Config instance in order to pass updates to it
        Config *_config;
};

#endif // CONFIGUPDATER_H
