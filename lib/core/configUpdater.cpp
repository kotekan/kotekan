#include "configUpdater.hpp"
#include "errors.h"
#include "restServer.hpp"

#include <iostream>
#include <boost/tokenizer.hpp>

configUpdater& configUpdater::instance()
{
    static configUpdater dm;

    return dm;
}

void configUpdater::apply_config(Config& config)
{
    _config = &config;

    // parse the tree and create endpoints
    parse_tree(config.get_full_config_json(), "");
}

void configUpdater::parse_tree(json& config_tree, const std::string& path)
{
    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it)
    {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_update_endpoint block, and if so create
        // the endpoint
        string endpoint_type = it.value().value("kotekan_update_endpoint",
                                                "none");
        string unique_name = path + "/" + it.key();
        if (endpoint_type == "json") {
            if (std::count(_endpoints.begin(), _endpoints.end(), unique_name)
                != 0) {
                throw std::runtime_error("configUpdater: An endpoint with the" \
                                         "path " + unique_name + " has been " \
                                         "defined more than once.");
            }
            INFO("configUpdater: creating endpoint: %s", unique_name.c_str());
            create_endpoint(unique_name);

            // Store initial value for a first update on subscription
            _init_values.insert(std::pair<std::string, nlohmann::json>(
                                   unique_name, it.value()));

            continue; // no recursive updatable blocks allowed
        }
        else if (endpoint_type != "none") {
            throw std::runtime_error("configUpdater: Found an unknown " \
                                     "endpoint type value: " + endpoint_type);
            continue; // no recursive updatable blocks allowed
        }

        // Recursive part.
        // This is a section/scope not a process block.
        parse_tree(it.value(), unique_name);
    }
}

void configUpdater::subscribe(const std::string& name,
                              std::function<bool(json &)> callback)
{
    _callbacks.insert(std::pair<std::string, std::function<bool(
                          nlohmann::json &)>>(name, callback));
    DEBUG("New subscription to %s", name.c_str());

    // First call to subscriber with initial value from the config
    if (!callback(_init_values[name])) {
        WARN("configUpdater: Failure when calling subscriber to set initial " \
             "value.");
        WARN("configUpdater: Stopping Kotekan.");
        raise(SIGINT);
    }
}

void configUpdater::create_endpoint(const string& name)
{
    // register POST endpoint
    // this will add any missing / in the beginning of the name
    restServer::instance().register_post_callback(name,
                     std::bind(&configUpdater::rest_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
    _endpoints.push_back(name);
}

void configUpdater::rest_callback(connectionInstance &con, nlohmann::json &json)
{
    DEBUG("Callback received this: %s", con.get_full_message().c_str());

    std::string uri = con.get_uri();

    DEBUG("uri called: %s", uri.c_str());

    // Call subscriber callbacks
    auto search = _callbacks.equal_range(uri);
    if (search.first == _callbacks.end()) {
        INFO("configUpdater: Received a POST command to endpoint %s, but " \
             "there are no subscribers to the endpoint.", uri.c_str());
        if (std::find(_endpoints.begin(), _endpoints.end(), uri)
            == _endpoints.end()) {
            WARN("configUpdater: Received POST command to non-existend " \
                 "endpoint: %s. This should never happen.", uri.c_str());
            return;
        }
    }
    while (search.first != search.second) {
        DEBUG("configUpdater: Calling subscriber of %s",
             search.first->first.c_str());

        // subscriber callback
        if (!search.first->second(json)) {
            con.send_empty_reply(HTTP_RESPONSE::INTERNAL_ERROR);
            WARN("configUpdater: Failed updating %s with value %s.",
                 uri.c_str(),
                 json.dump().c_str());
            WARN("Stopping Kotekan.");
            // Shut Kotekan down
            raise(SIGINT);
            break;
        }
        search.first++;
    }

    // update active configs with all values in this update
    for (nlohmann::json::iterator it = json.begin(); it != json.end(); it++) {
        DEBUG("configUpdater: Updating value %s with %s",
              std::string(uri + "/" + it.key()).c_str(),
              it.value().dump().c_str());
        // this ignores the data type, should be checked in processes' callbacks
        _config->update_value(uri, it.key(), it.value());
    }

    con.send_empty_reply(HTTP_RESPONSE::OK);
}
