#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include "Config.hpp"
#include "metadata.h"

#include "json.hpp"

#include <map>
#include <string>

namespace kotekan {

class metadataFactory {

public:
    // One metadataFactory should be created for each set of config and buffer_container
    metadataFactory(Config& config);
    ~metadataFactory();

    std::map<std::string, struct metadataPool*> build_pools();

private:
    void build_from_tree(std::map<string, struct metadataPool*>& pools, nlohmann::json& config_tree,
                         const std::string& path);
    struct metadataPool* new_pool(const std::string& pool_type, const std::string& location);

    Config& config;
};

} // namespace kotekan

#endif /* METADATA_FACTORY_HPP */
