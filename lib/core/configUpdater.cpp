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

void configUpdater::apply_config(Config& config, const string unique_name)
{
    _name = unique_name;
    DEBUG("configUpdater: reading dynamic block: %s", _name.c_str());

    // dynamic block has to be on root level
    vector<string> dynamic_blk = config.get_string_array("", unique_name);

    // parse config for subscribers to the endpoints
    //collect_receivers(config.get_full_config_json(), dynamic_blk);

    // FIXME: check if block present
    for (auto name : dynamic_blk)
    {
        // create endpoint
        DEBUG("creating endpoint: /%s", name.c_str());
        create_endpoint(name);
    }

    //TEST remove the next 2 lines
    _callbacks.insert(std::pair<std::string, int>("flags", 1));
    _callbacks.insert(std::pair<std::string, int>("gains", 2));

}

//void configUpdater::collect_receivers(json& config, vector<string>& endpoints)
//{
//}

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
    restServer::instance().register_post_callback(_name + "/" + name,
                     std::bind(&configUpdater::rest_callback, this,
                     std::placeholders::_1, std::placeholders::_2));
}

void configUpdater::rest_callback(connectionInstance &con, nlohmann::json &json)
{
    DEBUG("Callback received this: %s", con.get_full_message().c_str());

    std::string name = attribute_name(con);

    DEBUG("Found name: %s", name.c_str());

    // Call subscriber callbacks
    // TODO: call multiple subscribers (search++ ?)
    auto search = _callbacks.find(name);
    if (search != _callbacks.end()) {
        DEBUG("Calling subscriber: %s%d", search->first.c_str(), search->second)
    } else {
        DEBUG("Not found");
    }
}

std::string configUpdater::attribute_name(connectionInstance &con)
{
    std::string uri = con.get_uri();
    boost::char_separator<char> sep{"/"};
    boost::tokenizer<boost::char_separator<char>> tok{uri, sep};
    boost::tokenizer<boost::char_separator<char>>::iterator t = tok.begin();
//FIXME: test this
    if (t == tok.end())
        WARN("configUpdater: Failure parsing endpoint " \
             "name %s for attribute %s: Bad endpoint name.",
             uri, _name);
    if (*t != _name)
        WARN("configUpdater: Failure parsing endpoint " \
             "name %s for attribute %s: " \
             "Name of dynamic attribute block not found" \
             "in endpoint name.",
             uri, _name);

    t++;
    if (t == tok.end())
        WARN("configUpdater: Failure parsing endpoint " \
             "name %s for attribute %s: Attribute name not found" \
             "in endpoint name.", uri, _name);

    return *t;
}

