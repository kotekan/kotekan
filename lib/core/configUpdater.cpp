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
    // parse the tree and create endpoints
    parse_tree(config.get_full_config_json(), "");

//    // parse config for subscribers to the endpoints
//    //collect_receivers(config.get_full_config_json(), dynamic_blk);

//    // FIXME: check if block present
//    for (auto name : dynamic_blk)
//    {
//        // create endpoint

//        create_endpoint(name);
//    }

    //TEST remove the next 2 lines
    _callbacks.insert(std::pair<std::string, int>("/dynamic_attributes/flagging", 1));
    _callbacks.insert(std::pair<std::string, int>("gains", 2));
    _callbacks.insert(std::pair<std::string, int>("/dynamic_attributes/flagging", 2));


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
            DEBUG("configUpdater: creating endpoint: %s", unique_name.c_str());
            create_endpoint(unique_name);
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

void configUpdater::subscribe(const string& name,
                              std::function<bool(json &)> callback)
{
    //_callbacks.insert(std::pair<std::string,
      //                          std::function<void(connectionInstance &,
        //                        json &)>>(name, callback));
    _callbacks.insert(std::pair<std::string, int>(name, 1));
    DEBUG("New subscription to %s: %d", name, 1);
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
            == _endpoints.end())
            WARN("configUpdater: Received POST command to non-existend " \
                 "endpoint: %s. This should never happen.", uri.c_str());
    } else {
        while (search.first != search.second) {
            DEBUG("configUpdater: Calling subscriber: %s%d",
                  search.first->first.c_str(), search.first->second);
            //TODO actually call it
            search.first++;
        }
    }
}
