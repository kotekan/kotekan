#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include <string>
#include <map>

#include "json.hpp"
#include "metadata.h"
#include "Config.hpp"

// Name space includes.
using json = nlohmann::json;
using std::string;
using std::map;

class metadataFactory {

public:
    // One processFactory should be created for each set of config and buffer_container
    metadataFactory(Config &config);
    ~metadataFactory();

    map<string, struct metadataPool *> build_pools();

private:
    void build_from_tree(map<string, struct metadataPool *> &pools, json &config_tree, const string &path);
    struct metadataPool * new_pool(const string &pool_type, const string &location);

    Config &config;
};

#endif /* METADATA_FACTORY_HPP */