#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include "Config.hpp"
#include "metadata.h"

#include "json.hpp"

#include <map>
#include <string>

// Name space includes.
using json = nlohmann::json;
using std::map;
using std::string;

namespace kotekan {

class metadataFactory {

public:
    // One StageFactory should be created for each set of config and buffer_container
    metadataFactory(Config& config);
    ~metadataFactory();

    map<string, struct metadataPool*> build_pools();

private:
    void build_from_tree(map<string, struct metadataPool*>& pools, json& config_tree,
                         const string& path);
    struct metadataPool* new_pool(const string& pool_type, const string& location);

    Config& config;
};

} // namespace kotekan

#endif /* METADATA_FACTORY_HPP */
