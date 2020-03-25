#include "configUpdater.hpp"

#include "Config.hpp"         // for Config
#include "Stage.hpp"          // for Stage
#include "kotekanLogging.hpp" // for WARN_NON_OO, DEBUG_NON_OO, INFO_NON_OO, ERROR_NON_OO, FATA...
#include "restServer.hpp"     // for connectionInstance, HTTP_RESPONSE, HTTP_RESPONSE::BAD_REQUEST
#include "visUtil.hpp"        // for json_type_name

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, basic_json<>::iterator, basic_json, iter_impl, basic...

#include <algorithm> // for find, count
#include <exception> // for exception
#include <stdexcept> // for runtime_error
#include <utility>   // for pair

using nlohmann::json;

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
        INFO_NON_OO("configUpdater: Removing endpoint {:s}", *it);
        restServer::instance().remove_json_callback(*it);
    }

    // clear all memory of endpoints, callbacks and updatable blocks
    _endpoints.clear();
    _callbacks.clear();
    _init_values.clear();
    _keys.clear();
}

void configUpdater::parse_tree(const nlohmann::json& config_tree, const std::string& path) {
    for (json::const_iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_update_endpoint block, and if so create
        // the endpoint
        std::string endpoint_type = it.value().value("kotekan_update_endpoint", "none");
        std::string unique_name = fmt::format(fmt("{:s}/{:s}"), path, it.key());
        if (endpoint_type == "json") {
            if (std::count(_endpoints.begin(), _endpoints.end(), unique_name) != 0) {
                throw std::runtime_error(fmt::format(fmt("configUpdater: An endpoint with the path "
                                                         "{:s} has been defined more than once."),
                                                     unique_name));
            }
            INFO_NON_OO("configUpdater: creating endpoint: {:s}", unique_name);
            create_endpoint(unique_name);

            // Store initial values for a first update on subscription
            _init_values.insert(std::pair<std::string, nlohmann::json>(unique_name, it.value()));

            // Store all keys of this updatable block
            std::vector<std::string> keys;
            for (json::const_iterator key = it.value().begin(); key != it.value().end(); key++) {
                if (key.value().dump() != "kotekan_update_endpoint")
                    keys.push_back(key.key());
            }
            _keys.insert(std::pair<std::string, std::vector<std::string>>(unique_name, keys));

            continue; // no recursive updatable blocks allowed
        } else if (endpoint_type != "none") {
            throw std::runtime_error("configUpdater: Found an unknown endpoint type value: "
                                     + endpoint_type);
        }

        // Recursive part.
        // This is a section/scope not a stage block.
        parse_tree(it.value(), unique_name);
    }
}

void configUpdater::subscribe(const Stage* subscriber, std::function<bool(json&)> callback) {
    subscribe(_config->get<std::string>(subscriber->get_unique_name(), "updatable_config"),
              callback);
}

void configUpdater::subscribe(const Stage* subscriber,
                              std::map<std::string, std::function<bool(json&)>> callbacks) {

    // If no callbacks are passed, then we don't need to find any updatable_config blocks
    if (callbacks.size() == 0) return;

    // Find the nearest updatable config block
    auto updatable_config_paths = _config->get<std::map<std::string, std::string>>(
        subscriber->get_unique_name(), "updatable_config");

    // Extract the paths from that config block, it must contain all callbacks
    for (auto callback : callbacks) {
        std::string path;
        try {
            path = updatable_config_paths.at(callback.first);
        } catch (std::out_of_range) {
            throw std::runtime_error(
                fmt::format(fmt("The config option '{:s}' is required, but was not found in the "
                                "path: {:s}/updatable_config/"),
                            callback.first, subscriber->get_unique_name()));
        }
        subscribe(path, callback.second);
    }
}

void configUpdater::subscribe(const std::string& name, std::function<bool(json&)> callback) {
    if (!callback)
        throw std::runtime_error("configUpdater: Was passed a callback function for endpoint '"
                                 + name + "', that does not exist.");
    _callbacks.insert(std::pair<std::string, std::function<bool(nlohmann::json&)>>(name, callback));
    DEBUG_NON_OO("New subscription to {:s}", name);

    // First call to subscriber with initial value from the config
    if (!callback(_init_values[name]))
        throw std::runtime_error(
            "configUpdater: Failure when calling subscriber to set initial value at endpoint '"
            + name + "'.");
}

void configUpdater::create_endpoint(const std::string& name) {
    // register POST endpoint
    // this will add any missing / in the beginning of the name
    restServer::instance().register_post_callback(name, std::bind(&configUpdater::rest_callback,
                                                                  this, std::placeholders::_1,
                                                                  std::placeholders::_2));
    _endpoints.push_back(name);
}

void configUpdater::rest_callback(connectionInstance& con, nlohmann::json& json) {
    std::string uri = con.get_uri();
    DEBUG_NON_OO("configUpdater: received message on endpoint: {:s}", uri);

    // Check the incoming json for extra values
    for (nlohmann::json::iterator it = json.begin(); it != json.end(); it++) {
        if (std::find(_keys[uri].begin(), _keys[uri].end(), it.key()) == _keys[uri].end()) {
            // this key is not in the config file
            std::string msg = fmt::format(fmt("configUpdater: Update to endpoint '{:s}' contained "
                                              "value '{:s}' not defined in the updatable config "
                                              "block."),
                                          uri, it.key());
            WARN_NON_OO("{:s}", msg);
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
                std::string msg = fmt::format(fmt("configUpdater: Update to endpoint '{:s}' "
                                                  "contained value '{:s}' of type {:s} (expected "
                                                  "type {:s})."),
                                              uri, it.key(), json_type_name(it.value()),
                                              json_type_name(_init_values[uri].at(it.key())));

                WARN_NON_OO("{:s}", msg);
                con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
        }
    }
    // ...and for missing values
    for (auto it = _keys[uri].begin(); it != _keys[uri].end(); it++) {
        if (*it != "kotekan_update_endpoint" && json.find(*it) == json.end()) {
            // this key is in the config file, but missing in the update
            std::string msg = fmt::format(fmt("configUpdater: Update to endpoint '{:s}' is "
                                              "missing value '{:s}' that is defined in the config "
                                              "file."),
                                          uri, *it);
            WARN_NON_OO("{:s}", msg);
            con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    // Call subscriber callbacks
    auto search = _callbacks.equal_range(uri);
    if (search.first == _callbacks.end()) {
        INFO_NON_OO("configUpdater: Received a POST command to endpoint {:s}, but "
                    "there are no subscribers to the endpoint.",
                    uri);
        if (std::find(_endpoints.begin(), _endpoints.end(), uri) == _endpoints.end()) {
            std::string msg = fmt::format(fmt("configUpdater: Received POST command to "
                                              "non-existend endpoint: {:s}. This should never "
                                              "happen."),
                                          uri);
            WARN_NON_OO("{:s}", msg);
            con.send_error(msg, HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }
    while (search.first != search.second) {
        // subscriber callback
        if (!search.first->second(json)) {
            std::string msg = fmt::format(fmt("configUpdater: Failed updating {:s} with new "
                                              "values: {:s}."),
                                          uri, json.dump());
            ERROR_NON_OO("{:s}", msg);
            con.send_error(msg, HTTP_RESPONSE::INTERNAL_ERROR);
            FATAL_ERROR_NON_OO("configUpdater: Stopping Kotekan.");
            return;
        }
        search.first++;
    }

    // update active config with all values in this update
    for (nlohmann::json::iterator it = json.begin(); it != json.end(); it++) {
        DEBUG_NON_OO("configUpdater: Updating value {:s}/{:s} with {:s}", uri, it.key(),
                     it.value().dump());

        try {
            // this ignores the data type,
            // should be checked in stages' callbacks
            _config->update_value(uri, it.key(), it.value());
        } catch (const std::exception& e) {
            std::string msg = fmt::format(fmt("configUpdater: Failed applying update to endpoint "
                                              "{:s}: {:s}"),
                                          uri, e.what());
            ERROR_NON_OO("{:s}", msg);
            con.send_error(msg, HTTP_RESPONSE::INTERNAL_ERROR);
            FATAL_ERROR_NON_OO("configUpdater: Stopping Kotekan.");
            return;
        }
    }

    con.send_empty_reply(HTTP_RESPONSE::OK);
}

} // namespace kotekan
