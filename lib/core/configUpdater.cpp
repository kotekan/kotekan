#include "configUpdater.hpp"

#include "Stage.hpp"
#include "errors.h"
#include "restServer.hpp"
#include "visUtil.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <iostream>
#include <signal.h>

namespace kotekan {

configUpdater& configUpdater::instance() {
    static configUpdater dm;

    return dm;
}

void configUpdater::apply_config(Config& config) {
    reset();

    _config = &config;

    // parse the tree and create endpoints
    parse_tree(config.get_full_config_json(), "");
}

void configUpdater::reset() {
    // unsubscribe from all REST endpoints
    for (auto it = _endpoints.cbegin(); it != _endpoints.cend(); it++) {
        INFO("configUpdater: Removing endpoint %s", it->c_str());
        restServer::instance().remove_json_callback(*it);
    }

    // clear all memory of endpoints, callbacks and updatable blocks
    _endpoints.clear();
    _callbacks.clear();
    _init_values.clear();
    _keys.clear();
}

void configUpdater::parse_tree(json& config_tree, const std::string& path) {
    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_update_endpoint block, and if so create
        // the endpoint
        string endpoint_type = it.value().value("kotekan_update_endpoint", "none");
        string unique_name = path + "/" + it.key();
        if (endpoint_type == "json") {
            if (std::count(_endpoints.begin(), _endpoints.end(), unique_name) != 0) {
                throw std::runtime_error("configUpdater: An endpoint with the"
                                         "path "
                                         + unique_name
                                         + " has been "
                                           "defined more than once.");
            }
            INFO("configUpdater: creating endpoint: %s", unique_name.c_str());
            create_endpoint(unique_name);

            // Store initial values for a first update on subscription
            _init_values.insert(std::pair<std::string, nlohmann::json>(unique_name, it.value()));

            // Store all keys of this updatable block
            std::vector<std::string> keys;
            for (json::iterator key = it.value().begin(); key != it.value().end(); key++) {
                if (key.value().dump() != "kotekan_update_endpoint")
                    keys.push_back(key.key());
            }
            _keys.insert(std::pair<std::string, std::vector<std::string>>(unique_name, keys));

            continue; // no recursive updatable blocks allowed
        } else if (endpoint_type != "none") {
            throw std::runtime_error("configUpdater: Found an unknown "
                                     "endpoint type value: "
                                     + endpoint_type);
            continue; // no recursive updatable blocks allowed
        }

        // Recursive part.
        // This is a section/scope not a stage block.
        parse_tree(it.value(), unique_name);
    }
}

void configUpdater::subscribe(const Stage* subscriber, std::function<bool(json&)> callback) {
    if (!_config->exists(subscriber->get_unique_name(), "updatable_config"))
        throw std::runtime_error("configUpdater: key 'updatable_config' was "
                                 "not found in '"
                                 + subscriber->get_unique_name() + "' in the config file.");
    subscribe(_config->get<std::string>(subscriber->get_unique_name(), "updatable_config"),
              callback);
}

void configUpdater::subscribe(const Stage* subscriber,
                              std::map<std::string, std::function<bool(json&)>> callbacks) {
    for (auto callback : callbacks) {
        if (!_config->exists(subscriber->get_unique_name() + "/updatable_config", callback.first))
            throw std::runtime_error("configUpdater: key '" + callback.first
                                     + "' was not found in '" + subscriber->get_unique_name()
                                     + "/updatable_config' in the config file.");
        subscribe(_config->get<std::string>(subscriber->get_unique_name() + "/updatable_config/",
                                            callback.first),
                  callback.second);
    }
}

void configUpdater::subscribe(const std::string& name, std::function<bool(json&)> callback) {
    if (!callback)
        throw std::runtime_error("configUpdater: Was passed a callback "
                                 "function for endpoint '"
                                 + name
                                 + "', that "
                                   "does not exist.");
    _callbacks.insert(std::pair<std::string, std::function<bool(nlohmann::json&)>>(name, callback));
    DEBUG("New subscription to %s", name.c_str());

    // First call to subscriber with initial value from the config
    if (!callback(_init_values[name]))
        throw std::runtime_error("configUpdater: Failure when calling "
                                 "subscriber to set initial value at endpoint"
                                 " '"
                                 + name + "'.");
}

void configUpdater::create_endpoint(const string& name) {
    // register POST endpoint
    // this will add any missing / in the beginning of the name
    restServer::instance().register_post_callback(name, std::bind(&configUpdater::rest_callback,
                                                                  this, std::placeholders::_1,
                                                                  std::placeholders::_2));
    _endpoints.push_back(name);
}

void configUpdater::rest_callback(connectionInstance& con, nlohmann::json& json) {
    std::string uri = con.get_uri();
    DEBUG("configUpdater: received message on endpoint: %s", uri.c_str());

    // Check the incoming json for extra values
    for (nlohmann::json::iterator it = json.begin(); it != json.end(); it++) {
        if (std::find(_keys[uri].begin(), _keys[uri].end(), it.key()) == _keys[uri].end()) {
            // this key is not in the config file
            std::string msg = fmt::format("configUpdater: Update to endpoint "
                                          "'{}' contained value '{}' not "
                                          "defined in the updatable config "
                                          "block.",
                                          uri.c_str(), it.key().c_str());
            WARN(msg.c_str());
            con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
            return;
        } else {
            // Reject changes in general type (string vs. number vs. object),
            // and reject changes between (un)signed integers and floats, but allow both
            // signed and unsigned integers to be interchanged.
            if (it.value().type_name() != _init_values[uri].at(it.key()).type_name()
                || ((it.value().type() == nlohmann::json::value_t::number_float)
                    ^ (_init_values[uri].at(it.key()).type()
                       == nlohmann::json::value_t::number_float))) {
                std::string msg =
                    fmt::format("configUpdater: Update to endpoint '{}' contained"
                                " value '{}' of type {} (expected type {}).",
                                uri.c_str(), it.key().c_str(), json_type_name(it.value()),
                                json_type_name(_init_values[uri].at(it.key())));

                WARN(msg.c_str());
                con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
        }
    }
    // ...and for missing values
    for (auto it = _keys[uri].begin(); it != _keys[uri].end(); it++) {
        if (*it != "kotekan_update_endpoint" && json.find(*it) == json.end()) {
            // this key is in the config file, but missing in the update
            std::string msg = fmt::format("configUpdater: Update to endpoint "
                                          "'{}' is missing value '{}' that is"
                                          " defined in the config file.",
                                          uri.c_str(), it->c_str());
            WARN(msg.c_str());
            con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    // Call subscriber callbacks
    auto search = _callbacks.equal_range(uri);
    if (search.first == _callbacks.end()) {
        INFO("configUpdater: Received a POST command to endpoint %s, but "
             "there are no subscribers to the endpoint.",
             uri.c_str());
        if (std::find(_endpoints.begin(), _endpoints.end(), uri) == _endpoints.end()) {
            std::string msg = fmt::format("configUpdater: Received POST "
                                          "command to non-existend endpoint: "
                                          "%s. This should never happen.",
                                          uri.c_str());
            WARN(msg.c_str());
            con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }
    while (search.first != search.second) {
        // subscriber callback
        if (!search.first->second(json)) {
            std::string msg = fmt::format("configUpdater: Failed updating %s "
                                          "with value %s.",
                                          uri.c_str(), json.dump().c_str());
            ERROR(msg.c_str());
            con.send_error(msg, HTTP_RESPONSE::INTERNAL_ERROR);
            ERROR("configUpdater: Stopping Kotekan.");
            raise(SIGINT);
            return;
        }
        search.first++;
    }

    // update active config with all values in this update
    for (nlohmann::json::iterator it = json.begin(); it != json.end(); it++) {
        DEBUG("configUpdater: Updating value %s with %s", std::string(uri + "/" + it.key()).c_str(),
              it.value().dump().c_str());

        try {
            // this ignores the data type,
            // should be checked in stages' callbacks
            _config->update_value(uri, it.key(), it.value());
        } catch (const std::exception& e) {
            std::string msg = fmt::format("configUpdater: Failed applying "
                                          "update to endpoint %s: %s",
                                          uri.c_str(), e.what());
            ERROR(msg.c_str());
            con.send_error(msg, HTTP_RESPONSE::INTERNAL_ERROR);
            ERROR("configUpdater: Stopping Kotekan.");
            raise(SIGINT);
            return;
        }
    }

    con.send_empty_reply(HTTP_RESPONSE::OK);
}

} // namespace kotekan
