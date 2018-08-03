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
       static configUpdater& get();

       // Remove the implicit copy/assignments to prevent copying
       configUpdater(const configUpdater&) = delete;
       void operator=(const configUpdater&) = delete;

       /**
        * @brief Set and apply the static config to configUpdater
        * @param config         The config.
        * @param unique_name    The name of the dynamic block.
        */
       void apply_config(Config& config, const string unique_name);

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

       /**
        * @brief configUpdater::rest_callback
        * @param con
        * @param json
        */
       void rest_callback(connectionInstance &con,
                                         nlohmann::json &json);

    private:
        /// Constructor, we don't want anyone to call this
        configUpdater() { }

        /// Creates a new endpoint with a given name
        void create_endpoint(const string& name);

        /// Parses a POST message for the attribute name, where the endpoint is
        /// called /<name of dynamic block>/<attribute name>
        std::string attribute_name(connectionInstance &con);

        // name of the dynamic block in the config
        string _name;

        // mmap of all subscriber callback functions for the registered dynamic
        // attributes
        //std::multimap<std::string, std::function<void(connectionInstance &,
          //                                            json &)>> _callbacks;
        std::multimap<std::string, int> _callbacks;
};

#endif // CONFIGUPDATER_H
