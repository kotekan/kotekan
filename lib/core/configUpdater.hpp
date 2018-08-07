#ifndef CONFIGUPDATER_H
#define CONFIGUPDATER_H

#include "Config.hpp"
#include "restServer.hpp"


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
